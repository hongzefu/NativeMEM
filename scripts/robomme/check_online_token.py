"""在线 / 离线记忆 token 对照（1116 计划 C2 第 4 行、G9a / G9b、J#8 #15）。

两道判定，帧身份冻结在 ``scripts/robomme/online_token_frames.json``：

G9a ``ENCODER_IMPL``（硬闸门）：同一批 224 像素（每集 H5 的 front_jpeg / wrist_jpeg 经 S1 ``offline_frame`` 解码），
  分别走 ①服务器编码路径（``serve_nativemem.NativeMemServePolicy._encode_one``：``online_frame`` → 逐帧逐路
  batch=1 的 ``encode_frames``）与 ②离线抽取路径（直接调用 S1 ``extract_mem_tokens.encode_episode_chunk``：
  该帧所在的「本集第 64k 帧起 64 帧一批、末批 edge 补齐」，与离线抽取同一条代码路径），要求
  max_abs <= 1e-5 且逐 token 范数比在 [0.999, 1.001]。排除权重、尺度、视角错位等实现错误。
  给了 ``--memtokens`` 时另记录离线重算与 memmap 行的差（只记录，完整性由 G7 负责）。

G9b ``ONLINE_TOKEN=RECORDED``（只记录，用户裁决「只记录不拦截」）：同一帧的 raw 在线预处理
  （256 原图 → ``online_frame``，无 JPEG）与离线 JPEG 预处理（``resize_with_pad_cv2`` 224 + JPEG q95 → ``offline_frame``）
  得到的 token 之差，按「来源（exec＝V17 pkl 执行帧 / binfill_prefix＝BinFill 演示前缀库帧）× 视角」分层统计
  max_abs 与 cos_min。BinFill 前缀库的腕部帧本身是 JPEG q90（X13），照实记录。

用法：
  check_online_token.py freeze  [--out scripts/robomme/online_token_frames.json]   生成冻结帧身份（只读 policy 仓）
  check_online_token.py run     [--frames …] [--data-root …] [--memtokens …] [--max-exec-frames N] [--g9a-only] --report <json>
  正式 G9a/G9b 由主会话在 GPU 上起（第 5 步，G7 之后）；``--max-exec-frames 2 --g9a-only`` 可在 CPU 上做小样自检。
  run 先回校冻结 JSON 里的 ``episode_manifest_sha256`` 与当前 episode_manifest 的 ``sha256`` 字段，不一致即报错。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import pickle
import sys
from typing import Callable

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FORK_SRC = HERE.parents[1]
POLICY_SRC = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
V17_DATASET = POLICY_SRC / "v1-store/datasets/4task-v2-1600ep-604f16da"
EPISODE_MANIFEST = V17_DATASET / "meta/episode_manifest.json"
PKL_DIR = V17_DATASET / "source/data"
DEMO_STORE = POLICY_SRC / "v1-store/demo-prefix/binfill-3b4de03a0b46"
NM_ROOT = pathlib.Path("/scratch/hongze/NativeMEM")
DEFAULT_DATA_ROOT = NM_ROOT / "data/robomme4task-v2"
DEFAULT_ENCODER = NM_ROOT / "checkpoints/mem_tokenizer_pretrain/pretrain/49999/params"
DEFAULT_FRAMES = HERE / "online_token_frames.json"
VIEWS = ("front", "wrist")
TASK_OF_H5 = {"record_dataset_BinFill.h5": "BinFill", "record_dataset_RouteStick.h5": "RouteStick",
              "record_dataset_VideoRepick.h5": "VideoRepick", "record_dataset_VideoUnmaskSwap.h5": "VideoUnmaskSwap"}
G9A_MAX_ABS = 1e-5
G9A_RATIO = (0.999, 1.001)


def _load(name: str, path: pathlib.Path):
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


# ── 冻结帧身份 ────────────────────────────────────────────────────────────

def freeze(out: pathlib.Path) -> dict:
    """每任务取 4 集（按 raw_ep_idx 的 0/133/266/399 号）× 2 帧（首个执行帧、末帧）= 32 个执行帧；
    BinFill 前缀库按 key 排序取 easy/medium/hard 各前 1 条加 hard 第 2 条，每条取第 0 帧与第 D-1 帧 = 8 帧。
    逐帧读 V17 pkl 核对 epis_idx/step_idx（证明 row 映射正确），不写 policy 仓。"""
    manifest = json.loads(EPISODE_MANIFEST.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict]] = {}
    for e in manifest["episodes"]:
        by_task.setdefault(TASK_OF_H5[e["h5_file"]], []).append(e)
    exec_frames = []
    for task in sorted(by_task):
        eps = sorted(by_task[task], key=lambda e: e["raw_ep_idx"])
        for pick in (0, 133, 266, 399):
            e = eps[pick]
            for t in (e["exec_start_idx"], e["num_timesteps"] - 1):
                pkl_index = e["exec_sample_offset"] + (t - e["exec_start_idx"])
                with open(PKL_DIR / f"{pkl_index}.pkl", "rb") as f:
                    s = pickle.load(f)
                if int(s["epis_idx"][0]) != e["global_episode_idx"] or int(s["step_idx"][0]) != t:
                    raise ValueError(f"pkl {pkl_index} 指向 epis={s['epis_idx']} step={s['step_idx']}，期望 "
                                     f"{e['global_episode_idx']}/{t}")
                exec_frames.append({"task": task, "global_episode_idx": e["global_episode_idx"],
                                    "raw_ep_idx": e["raw_ep_idx"], "t": t, "pkl_index": pkl_index,
                                    "row": e["total_sample_offset"] + t,
                                    "position": "exec_first" if t == e["exec_start_idx"] else "episode_last"})
    index = json.loads((DEMO_STORE / "index.json").read_text(encoding="utf-8"))
    ok = sorted(k for k, v in index.items() if v.get("demo_status") == "ok")
    picks = []
    for diff in ("easy", "medium", "hard"):
        picks.append(next(k for k in ok if k.startswith(f"BinFill/{diff}/")))
    picks.append([k for k in ok if k.startswith("BinFill/hard/")][1])
    prefix_frames = []
    for key in picks:
        d = int(index[key]["D"])
        for i in (0, d - 1):
            prefix_frames.append({"key": key, "frame": i, "D": d})
    payload = {
        "version": 1,
        "purpose": "G9a ENCODER_IMPL / G9b ONLINE_TOKEN 冻结帧身份（1116 计划 J#8）",
        "view_order": list(VIEWS),
        "row_rule": "row = episode_manifest.episodes[e].total_sample_offset + t（t 含演示帧）",
        "sources": {"episode_manifest": str(EPISODE_MANIFEST), "pkl_dir": str(PKL_DIR),
                    "episode_manifest_sha256": manifest.get("sha256", ""), "demo_prefix_store": str(DEMO_STORE)},
        "note": ("V17 pkl 只含执行帧，非 BinFill 任务的演示帧没有离线 raw 原图可取；演示帧分层只由 BinFill 前缀库帧代表"
                 "（前视 PNG 无损、腕部 JPEG q90）。"),
        "exec_frames": exec_frames,
        "binfill_prefix_frames": prefix_frames,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"FROZEN exec_frames={len(exec_frames)} binfill_prefix_frames={len(prefix_frames)} out={out}", flush=True)
    return payload


# ── 统计 ──────────────────────────────────────────────────────────────────

def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a、b 形状 [N,1152]：max_abs、cos_min、逐 token 范数比 min/max。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-30)
    ratio = na / np.maximum(nb, 1e-30)
    return {"n": int(a.shape[0]), "max_abs": float(np.abs(a - b).max()), "cos_min": float(cos.min()),
            "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())}


def g9a_verdict(stats: dict) -> bool:
    return stats["max_abs"] <= G9A_MAX_ABS and G9A_RATIO[0] <= stats["ratio_min"] and stats["ratio_max"] <= G9A_RATIO[1]


def server_path_tokens(serve_mod, encode_fn: Callable, online_frame_fn: Callable, pairs: list) -> np.ndarray:
    """经服务器真实代码 NativeMemServePolicy._encode_one：每对 (front, wrist) → [2,1152]。返回 [N,2,1152]。"""
    sv = serve_mod.NativeMemServePolicy(None, encode_fn, online_frame_fn)
    out = []
    for i, (front, wrist) in enumerate(pairs):
        sv.raw_frames[i] = (front, wrist)
        out.append(sv._encode_one(i))
    return np.stack(out, axis=0)


def offline_chunk_tokens(xm, encode_fn: Callable, data_root: pathlib.Path, exec_frames: list) -> np.ndarray:
    """离线路径：直接调用 S1 ``extract_mem_tokens.encode_episode_chunk``（同一批次划分、同一补齐方式）。
    每个 (集, 64 帧批) 只编码一次；返回 [N,2,1152]，与 exec_frames 顺序一致。"""
    from concurrent.futures import ThreadPoolExecutor

    import h5py

    cache: dict[tuple[int, int], np.ndarray] = {}
    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fr in exec_frames:
            g, t = fr["global_episode_idx"], fr["t"]
            start = (t // xm.BATCH_SIZE) * xm.BATCH_SIZE
            if (g, start) not in cache:
                with h5py.File(pathlib.Path(data_root) / "episodes" / f"ep_{g:04d}.h5", "r") as f:
                    cache[(g, start)] = xm.encode_episode_chunk(encode_fn, f, start, pool)
            out.append(cache[(g, start)][t - start])
    return np.stack(out, axis=0).astype(np.float32)


def check_manifest_pin(frames: dict) -> str:
    """F5：冻结 JSON 记录的 episode_manifest sha256 必须等于当前 manifest 的 ``sha256`` 字段。"""
    want = frames["sources"]["episode_manifest_sha256"]
    got = json.loads(pathlib.Path(frames["sources"]["episode_manifest"]).read_text(encoding="utf-8")).get("sha256", "")
    if not want or got != want:
        raise SystemExit(f"MANIFEST_PIN=FAIL 冻结 {want[:12] or '缺失'} != 当前 {got[:12] or '缺失'}"
                         f"（{frames['sources']['episode_manifest']}），帧身份已失效，拒绝运行")
    print(f"MANIFEST_PIN=PASS sha256={got[:12]}", flush=True)
    return got


def run(args) -> int:
    import h5py

    from openpi.robomme import mem_encoder

    serve_mod = _load("nm_serve_nativemem", HERE / "serve_nativemem.py")
    xm = _load("nm_extract_mem_tokens", HERE / "extract_mem_tokens.py")
    prep = _load("nm_prepare_mem_tokenizer_h5", FORK_SRC / "scripts/prepare_mem_tokenizer_h5.py")
    demo_mod = _load("nm_client_demo_prefix", HERE / "client/demo_prefix.py")
    import cv2

    report_path = pathlib.Path(args.report).resolve()
    if str(report_path).startswith(str(POLICY_SRC.resolve()) + "/"):
        raise SystemExit(f"报告路径落在 policy 仓，拒绝：{report_path}")
    frames = json.loads(pathlib.Path(args.frames).read_text(encoding="utf-8"))
    check_manifest_pin(frames)
    exec_list = frames["exec_frames"][: args.max_exec_frames] if args.max_exec_frames else frames["exec_frames"]
    enc = mem_encoder.load_encoder(str(args.encoder_params))

    def encode_fn(x):
        return np.asarray(mem_encoder.encode_frames(enc, x))

    def jpeg_path(img: np.ndarray) -> np.ndarray:
        """离线 JPEG 预处理：resize_with_pad_cv2 224 + JPEG q95（原版 prepare_mem_tokenizer_h5）→ S1 offline_frame。"""
        ok, buf = cv2.imencode(".jpg", prep.resize_with_pad_cv2(img, 224, 224), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise ValueError("JPEG 编码失败")
        return mem_encoder.offline_frame(buf.reshape(-1).tobytes())

    # ── 收集：执行帧（pkl raw 256 + 每集 H5 的 224 JPEG）与前缀库帧 ──
    exec_raw, exec_off224, rows, tags = [], [], [], []
    for fr in exec_list:
        with open(PKL_DIR / f"{fr['pkl_index']}.pkl", "rb") as f:
            s = pickle.load(f)
        assert int(s["epis_idx"][0]) == fr["global_episode_idx"] and int(s["step_idx"][0]) == fr["t"], fr
        with h5py.File(pathlib.Path(args.data_root) / "episodes" / f"ep_{fr['global_episode_idx']:04d}.h5", "r") as h:
            off = tuple(mem_encoder.offline_frame(np.asarray(h[f"{v}_jpeg"][fr["t"]], dtype=np.uint8).tobytes())
                        for v in VIEWS)
        exec_raw.append((s["image"], s["wrist_image"]))
        exec_off224.append(off)
        rows.append(fr["row"])
        tags.append(f"{fr['task']}/{fr['global_episode_idx']}/{fr['t']}")
    store = demo_mod.DemoPrefixStore(str(DEMO_STORE)) if not args.g9a_only else None
    prefix_raw, cache = [], {}
    for fr in ([] if args.g9a_only else frames["binfill_prefix_frames"]):
        task, diff, ep = fr["key"].split("/")
        if fr["key"] not in cache:
            cache[fr["key"]] = store.get(task, diff, int(ep))
        entry = cache[fr["key"]]
        prefix_raw.append((entry["images"][fr["frame"]], entry["wrist_images"][fr["frame"]]))

    # ── G9a：同一批 224 像素，服务器路径 vs 离线路径 ──
    tok_server = server_path_tokens(serve_mod, encode_fn, mem_encoder.online_frame, exec_off224)
    tok_offline = offline_chunk_tokens(xm, encode_fn, pathlib.Path(args.data_root), exec_list)
    g9a = pair_stats(tok_server.reshape(-1, tok_server.shape[-1]), tok_offline.reshape(-1, tok_offline.shape[-1]))
    g9a_ok = g9a_verdict(g9a)
    report = {"g9a": g9a, "g9a_pass": g9a_ok, "frames": tags, "offline_batch": xm.BATCH_SIZE,
              "offline_path": "extract_mem_tokens.encode_episode_chunk", "jax_backend": __import__("jax").default_backend()}
    if args.memtokens:
        mm = np.memmap(args.memtokens, dtype=np.float32, mode="r").reshape(-1, 2, tok_offline.shape[-1])
        mem_rows = np.stack([np.asarray(mm[r]) for r in rows], axis=0)
        report["memmap_vs_offline"] = pair_stats(mem_rows.reshape(-1, mem_rows.shape[-1]),
                                                 tok_offline.reshape(-1, tok_offline.shape[-1]))
        print("MEMMAP_VS_OFFLINE " + json.dumps(report["memmap_vs_offline"]), flush=True)

    g9a_line = (f"ENCODER_IMPL={'PASS' if g9a_ok else 'FAIL'} n={len(exec_off224)} max_abs={g9a['max_abs']:.3g} "
                f"ratio_min={g9a['ratio_min']:.6f} ratio_max={g9a['ratio_max']:.6f} tokens={g9a['n']} "
                f"backend={report['jax_backend']}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.g9a_only:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(g9a_line, flush=True)
        return 0 if g9a_ok else 1

    # ── G9b：raw 在线 vs JPEG 离线，分层只记录 ──
    strata = {}
    exec_online = server_path_tokens(serve_mod, encode_fn, mem_encoder.online_frame, exec_raw)
    prefix_online = server_path_tokens(serve_mod, encode_fn, mem_encoder.online_frame, prefix_raw)
    prefix_jpeg = server_path_tokens(serve_mod, encode_fn, mem_encoder.online_frame,
                                     [(jpeg_path(f), jpeg_path(w)) for f, w in prefix_raw])
    for src, online, offline in (("exec", exec_online, tok_offline), ("binfill_prefix", prefix_online, prefix_jpeg)):
        for vi, view in enumerate(VIEWS):
            strata[f"{src}/{view}"] = pair_stats(online[:, vi], offline[:, vi])
            print(f"ONLINE_TOKEN_STRATUM {src}/{view} " + json.dumps(strata[f"{src}/{view}"]), flush=True)
    all_on = np.concatenate([exec_online, prefix_online]).reshape(-1, exec_online.shape[-1])
    all_off = np.concatenate([tok_offline, prefix_jpeg]).reshape(-1, exec_online.shape[-1])
    overall = pair_stats(all_on, all_off)
    report.update({"g9b_strata": strata, "g9b_overall": overall,
                   "n_exec_frames": len(exec_raw), "n_prefix_frames": len(prefix_raw)})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(g9a_line, flush=True)
    print(f"ONLINE_TOKEN=RECORDED n={len(exec_raw) + len(prefix_raw)} max_abs={overall['max_abs']:.4g} "
          f"cos_min={overall['cos_min']:.6f} report={report_path}", flush=True)
    return 0 if g9a_ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out", default=str(DEFAULT_FRAMES))
    r = sub.add_parser("run")
    r.add_argument("--frames", default=str(DEFAULT_FRAMES))
    r.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    r.add_argument("--memtokens", default="", help="mem_tokens.f32.bin（可选，只记录与离线重算的差）")
    r.add_argument("--encoder-params", default=str(DEFAULT_ENCODER))
    r.add_argument("--max-exec-frames", type=int, default=0, help="只取前 N 个执行帧（0=全部 32；CPU 小样自检用）")
    r.add_argument("--g9a-only", action="store_true", help="只跑 G9a，不做 G9b（CPU 小样自检用）")
    r.add_argument("--report", required=True, help="报告 json（落 fork runs/，不得在 policy 仓）")
    args = ap.parse_args()
    if args.cmd == "freeze":
        freeze(pathlib.Path(args.out))
    else:
        sys.exit(run(args))


if __name__ == "__main__":
    main()
