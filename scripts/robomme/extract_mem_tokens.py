"""Stage 1.5：用作者 Stage 1 记忆编码器把每集 H5 的每帧两路画面离线算成记忆 token（计划 C1、G7）。

照官方 examples/robotwin/convert_raw_to_lerobot_mem_tokenizer.py::MemTokenExtractor：
  离线输入 = 每集 H5 的 front_jpeg／wrist_jpeg（resize_with_pad_cv2 224 + JPEG q95）cv2.imdecode 解码，
  再 image_tools.resize_with_pad（224→224 恒等）/255·2−1 → SiglipEncoder → query 输出 x[:, :1]；
  按集从第 0 帧起每 64 帧一批，末批用最后一帧 edge 补齐到 64（与官方 BATCH_SIZE=64 相同，固定形状不重编译）。

输出 <data_root>/mem_tokens.f32.bin：np.memmap f32 [N_total, 2, 1152]，视角顺序 [front, wrist]，
行号 row = episode_manifest.episodes[e].total_sample_offset + t（t 为全帧号，含演示帧；不是执行帧 offset）。

三种模式：
  --init                               建空 memmap（稀疏文件）与 mem_tokens.meta.json（身份与分片计划）
  --num-shards 8 --shard-index i       跑第 i 片（一卡一片，由调用方设 CUDA_VISIBLE_DEVICES）；
                                       每集写完 flush 后更新进度，可断点续跑；全片完成写 shards/shard_<i>.done.json
  --finalize                           调 check_memtokens.py（G7 全部条件），全过才写 MEMTOKENS_DONE.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.robomme.dataset import NUM_VIEWS  # noqa: E402
from openpi.robomme.dataset import VIEW_ORDER  # noqa: E402
from openpi.robomme.dataset import episode_path  # noqa: E402
from openpi.robomme.dataset import load_manifest  # noqa: E402
from openpi.robomme.mem_encoder import TOKEN_DIM  # noqa: E402
from openpi.robomme.mem_encoder import offline_frame  # noqa: E402

BATCH_SIZE = 64  # 官方转换脚本 BATCH_SIZE
DEFAULT_DATA_ROOT = "/scratch/hongze/NativeMEM/data/robomme4task-v2"
DEFAULT_MANIFEST = (
    "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/"
    "4task-v2-1600ep-604f16da/meta/episode_manifest.json"
)
DEFAULT_PARAMS = "/scratch/hongze/NativeMEM/checkpoints/mem_tokenizer_pretrain/pretrain/49999/params"
MEMMAP_NAME = "mem_tokens.f32.bin"
META_NAME = "mem_tokens.meta.json"
DONE_NAME = "MEMTOKENS_DONE.json"
ROW_RULE = "row = episode_manifest.episodes[e].total_sample_offset + t (t = 全帧号，含演示帧)"
PREPROCESS = (
    "H5 front_jpeg/wrist_jpeg (resize_with_pad_cv2 224 + JPEG q95, RGB) -> cv2.imdecode(IMREAD_COLOR) 原样 -> "
    "image_tools.resize_with_pad(224,224) -> /255*2-1 -> SiglipEncoder So400m/14 float32 -> x[:, :1]; "
    "每集从第 0 帧起 64 帧一批，末批 edge 补齐"
)


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_plan(episodes: list[dict], num_shards: int) -> list[dict]:
    """按帧数把 episode 切成 num_shards 段连续区间（按 global 顺序），返回每片的 episode 与行区间。"""
    n = np.array([e["num_timesteps"] for e in episodes], dtype=np.int64)
    cum = np.cumsum(n)
    total = int(cum[-1])
    bounds = [0]
    for k in range(1, num_shards):
        target = total * k / num_shards
        b = int(np.searchsorted(cum, target, side="left")) + 1
        b = max(b, bounds[-1] + 1)
        bounds.append(min(b, len(episodes) - (num_shards - k)))
    bounds.append(len(episodes))
    plan = []
    for i in range(num_shards):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            raise ValueError(f"分片 {i} 为空")
        row_start = int(episodes[lo]["total_sample_offset"])
        row_end = int(episodes[hi - 1]["total_sample_offset"] + episodes[hi - 1]["num_timesteps"])
        plan.append({"shard": i, "ep_start": lo, "ep_end": hi, "row_start": row_start, "row_end": row_end})
    return plan


def h5_identity_digest(episodes_dir: pathlib.Path, episodes: list[dict]) -> tuple[str, int]:
    """源数据身份：按 global 顺序拼接每集 H5 attrs.h5_sha256（原始 H5 的 sha256）再取 sha256。"""
    h = hashlib.sha256()
    for e in episodes:
        with h5py.File(episode_path(episodes_dir, e["global_episode_idx"]), "r") as f:
            v = f.attrs["h5_sha256"]
            v = v.decode() if isinstance(v, bytes) else str(v)
            if int(f.attrs["num_timesteps"]) != int(e["num_timesteps"]):
                raise ValueError(f"ep {e['global_episode_idx']} num_timesteps 与 manifest 不符")
        h.update(f"{e['global_episode_idx']}:{v}\n".encode())
    return h.hexdigest(), len(episodes)


def open_memmap(path: pathlib.Path, rows: int, mode: str = "r") -> np.memmap:
    mm = np.memmap(path, dtype=np.float32, mode=mode)
    if mm.size != rows * NUM_VIEWS * TOKEN_DIM:
        raise ValueError(f"{path} 元素数 {mm.size} ≠ {rows}×{NUM_VIEWS}×{TOKEN_DIM}")
    return mm.reshape(rows, NUM_VIEWS, TOKEN_DIM)


def decode_views(f: h5py.File, start: int, stop: int, pool: ThreadPoolExecutor) -> tuple[np.ndarray, np.ndarray]:
    front = list(pool.map(offline_frame, f["front_jpeg"][start:stop]))
    wrist = list(pool.map(offline_frame, f["wrist_jpeg"][start:stop]))
    return np.stack(front), np.stack(wrist)


def encode_chunk(encode_fn, front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """一批（≤64 帧）两路编码，末批 edge 补齐到 64 再截断；返回 f32 [n, 2, 1152]。"""
    n = front.shape[0]
    if n < BATCH_SIZE:
        pad = ((0, BATCH_SIZE - n), (0, 0), (0, 0), (0, 0))
        front = np.pad(front, pad, mode="edge")
        wrist = np.pad(wrist, pad, mode="edge")
    tf = np.asarray(encode_fn(front), dtype=np.float32)[:n]
    tw = np.asarray(encode_fn(wrist), dtype=np.float32)[:n]
    return np.stack([tf, tw], axis=1)


def encode_episode_chunk(encode_fn, f: h5py.File, chunk_start: int, pool: ThreadPoolExecutor) -> np.ndarray:
    """本集第 chunk_start 帧起的一批（chunk_start 必须是 64 的倍数，与抽取时完全相同的批次划分）。"""
    if chunk_start % BATCH_SIZE:
        raise ValueError("chunk_start 必须是 64 的倍数")
    n = int(f.attrs["num_timesteps"])
    stop = min(chunk_start + BATCH_SIZE, n)
    front, wrist = decode_views(f, chunk_start, stop, pool)
    return encode_chunk(encode_fn, front, wrist)


def make_encode_fn(params_path: str):
    from openpi.robomme.mem_encoder import encode_frames
    from openpi.robomme.mem_encoder import load_encoder

    enc = load_encoder(params_path)
    return lambda frames: encode_frames(enc, frames)


def load_pin(pin_path: pathlib.Path) -> dict:
    pin = json.loads(pin_path.read_text())
    return {"revision": pin["revision"], "manifest_sha256": pin["manifest_sha256"], "repo_id": pin["repo_id"]}


def cmd_init(args) -> None:
    data_root = pathlib.Path(args.data_root)
    mm_path = data_root / MEMMAP_NAME
    meta_path = data_root / META_NAME
    if mm_path.exists() or meta_path.exists():
        raise SystemExit(f"MEMTOKENS_INIT=FAIL reason=exists path={mm_path}（拒绝覆盖，确认后由主会话手动处理旧文件）")
    episodes = load_manifest(args.manifest)
    rows = int(sum(e["num_timesteps"] for e in episodes))
    digest, n_ep = h5_identity_digest(data_root / "episodes", episodes)
    plan = shard_plan(episodes, args.num_shards)
    meta = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": rows,
        "views": NUM_VIEWS,
        "dim": TOKEN_DIM,
        "dtype": "float32",
        "view_order": list(VIEW_ORDER),
        "row_rule": ROW_RULE,
        "row_offset_field": "total_sample_offset",
        "episodes": n_ep,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "source_h5_sha256_digest": digest,
        "params_path": str(args.params_path),
        "tokenizer_pin": load_pin(pathlib.Path(args.pin)),
        "preprocess": PREPROCESS,
        "batch_size": BATCH_SIZE,
        "num_shards": args.num_shards,
        "shard_plan": plan,
    }
    (data_root / "shards").mkdir(parents=True, exist_ok=True)
    with open(mm_path, "wb") as f:
        f.truncate(rows * NUM_VIEWS * TOKEN_DIM * 4)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"MEMTOKENS_INIT=PASS rows={rows} episodes={n_ep} shards={args.num_shards} bytes={rows * NUM_VIEWS * TOKEN_DIM * 4}")


def run_shard(data_root: pathlib.Path, shard_index: int, encode_fn, *, log=print) -> dict:
    meta = json.loads((data_root / META_NAME).read_text())
    plan = meta["shard_plan"][shard_index]
    episodes = load_manifest(meta["manifest_path"])
    mm = open_memmap(data_root / MEMMAP_NAME, meta["rows"], mode="r+")
    shards_dir = data_root / "shards"
    progress_path = shards_dir / f"shard_{shard_index}.progress.json"
    done_path = shards_dir / f"shard_{shard_index}.done.json"
    if done_path.exists():
        log(f"[shard {shard_index}] 已完成，跳过")
        return json.loads(done_path.read_text())
    done_through = -1
    if progress_path.exists():
        done_through = int(json.loads(progress_path.read_text())["done_through_ep"])
        log(f"[shard {shard_index}] 断点续跑：已完成到 ep {done_through}")
    t0 = time.time()
    rows_done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ep in range(plan["ep_start"], plan["ep_end"]):
            if ep <= done_through:
                continue
            e = episodes[ep]
            off, n = int(e["total_sample_offset"]), int(e["num_timesteps"])
            with h5py.File(episode_path(data_root / "episodes", ep), "r") as f:
                if int(f.attrs["num_timesteps"]) != n or int(f.attrs["global_episode_idx"]) != ep:
                    raise ValueError(f"ep {ep} H5 attrs 与 manifest 不符")
                for start in range(0, n, BATCH_SIZE):
                    tokens = encode_episode_chunk(encode_fn, f, start, pool)
                    mm[off + start : off + start + tokens.shape[0]] = tokens
            mm.flush()
            rows_done += n
            progress_path.write_text(json.dumps({"done_through_ep": ep}) + "\n")
            if ep % 20 == 0:
                rate = rows_done / max(time.time() - t0, 1e-6)
                log(f"[shard {shard_index}] ep {ep}/{plan['ep_end'] - 1} rows_done={rows_done} {rate:.1f} 帧/s", flush=True)
    region = np.asarray(mm[plan["row_start"] : plan["row_end"]])
    nonfinite = int((~np.isfinite(region)).any(axis=(1, 2)).sum())
    zero_rows = int((np.abs(region).max(axis=2) == 0).any(axis=1).sum())
    done = {
        **plan,
        "rows": plan["row_end"] - plan["row_start"],
        "nonfinite_rows": nonfinite,
        "zero_rows": zero_rows,
        "elapsed_s": round(time.time() - t0, 1),
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    done_path.write_text(json.dumps(done, indent=2) + "\n")
    log(f"MEMTOKENS_SHARD=PASS shard={shard_index} rows={done['rows']} nonfinite={nonfinite} zero_rows={zero_rows}")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init", action="store_true")
    mode.add_argument("--shard-index", type=int)
    mode.add_argument("--finalize", action="store_true")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--params-path", default=DEFAULT_PARAMS)
    ap.add_argument("--pin", default=str(HERE / "tokenizer_pin.json"))
    args = ap.parse_args()

    if args.init:
        cmd_init(args)
    elif args.shard_index is not None:
        import jax

        print(f"[shard {args.shard_index}] jax.devices()={jax.devices()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        meta = json.loads((pathlib.Path(args.data_root) / META_NAME).read_text())
        if meta["num_shards"] != args.num_shards:
            raise SystemExit(f"MEMTOKENS_SHARD=FAIL reason=num_shards meta={meta['num_shards']} arg={args.num_shards}")
        run_shard(pathlib.Path(args.data_root), args.shard_index, make_encode_fn(meta["params_path"]))
    else:
        cmd = [
            sys.executable, str(HERE / "check_memtokens.py"),
            "--data-root", args.data_root, "--num-shards", str(args.num_shards), "--pin", args.pin,
        ]
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
