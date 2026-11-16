"""G7：记忆 token memmap 完整性检查（计划 E 表 G7、J#3）。全部通过才写 <data_root>/MEMTOKENS_DONE.json。

检查项：
  1. meta：view_order == [front, wrist]；行号规则用全帧 total_sample_offset；rows／dim 与 manifest 一致；
     manifest sha256 与 meta 记录一致；源 H5 身份摘要（逐集 attrs.h5_sha256）与 meta 一致；
     tokenizer 身份与 tokenizer_pin.json（revision + manifest_sha256）一致。
  2. 分片：num_shards 个 done 标记齐全；行区间与 meta 分片计划一致，排序后无缺口（gaps）无重叠（overlaps），
     覆盖 [0, rows)。
  3. 全表扫描：非有限值行数 nonfinite、任一视角全零的行数 zero_rows，必须都为 0。
  4. 独立重算：覆盖 4 任务 × 每任务首／末集 × {第 0 帧、演示末帧、执行首帧、末帧} 以及每个分片的首行／末行，
     按抽取时完全相同的 64 帧批次重算该帧所在整批，与表中各行逐位相等（np.array_equal）。
末行：MEMTOKENS=PASS|FAIL rows=… views=2 dim=1152 nonfinite=0 zero_rows=0 shards=8/8 gaps=0 overlaps=0 recompute=<k>/<n>
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extract_mem_tokens as ext  # noqa: E402

from openpi.robomme.dataset import NUM_VIEWS  # noqa: E402
from openpi.robomme.dataset import VIEW_ORDER  # noqa: E402
from openpi.robomme.dataset import episode_path  # noqa: E402
from openpi.robomme.dataset import load_manifest  # noqa: E402
from openpi.robomme.mem_encoder import TOKEN_DIM  # noqa: E402


def recompute_targets(episodes: list[dict], plan: list[dict]) -> list[tuple[int, int, str]]:
    """返回 (global_ep, 帧号, 标签) 列表，去重。"""
    by_task: dict[str, list[dict]] = {}
    for e in episodes:
        by_task.setdefault(e.get("h5_file", "all"), []).append(e)
    targets: dict[tuple[int, int], str] = {}
    for task, eps in sorted(by_task.items()):
        for which, e in (("first", eps[0]), ("last", eps[-1])):
            g, n, s = e["global_episode_idx"], e["num_timesteps"], e["exec_start_idx"]
            for label, t in (("f0", 0), ("demo_end", s - 1), ("exec_start", s), ("last", n - 1)):
                if 0 <= t < n:
                    targets.setdefault((g, t), f"{task}:{which}:{label}")
    for p in plan:
        first, last = episodes[p["ep_start"]], episodes[p["ep_end"] - 1]
        targets.setdefault((first["global_episode_idx"], 0), f"shard{p['shard']}:first_row")
        targets.setdefault((last["global_episode_idx"], last["num_timesteps"] - 1), f"shard{p['shard']}:last_row")
    return [(g, t, label) for (g, t), label in sorted(targets.items())]


def scan(mm: np.ndarray, chunk_rows: int = 65536) -> tuple[int, int]:
    nonfinite = zero_rows = 0
    for s in range(0, mm.shape[0], chunk_rows):
        block = np.asarray(mm[s : s + chunk_rows])
        nonfinite += int((~np.isfinite(block)).any(axis=(1, 2)).sum())
        zero_rows += int((np.abs(block).max(axis=2) == 0).any(axis=1).sum())
    return nonfinite, zero_rows


def check(
    data_root: str | pathlib.Path,
    num_shards: int,
    pin_path: str | pathlib.Path,
    encode_fn_factory=None,
    *,
    write_done: bool = True,
) -> tuple[bool, str]:
    data_root = pathlib.Path(data_root)
    reasons: list[str] = []
    meta = json.loads((data_root / ext.META_NAME).read_text())
    episodes = load_manifest(meta["manifest_path"])
    rows = int(sum(e["num_timesteps"] for e in episodes))

    # 1. meta 与身份
    if meta.get("view_order") != list(VIEW_ORDER):
        reasons.append("view_order")
    if meta.get("row_offset_field") != "total_sample_offset":
        reasons.append("row_offset_field")
    if meta.get("rows") != rows or meta.get("dim") != TOKEN_DIM or meta.get("views") != NUM_VIEWS:
        reasons.append("shape_meta")
    if ext.sha256_file(meta["manifest_path"]) != meta.get("manifest_sha256"):
        reasons.append("manifest_sha256")
    digest, _ = ext.h5_identity_digest(data_root / "episodes", episodes)
    if digest != meta.get("source_h5_sha256_digest"):
        reasons.append("source_h5_digest")
    if ext.load_pin(pathlib.Path(pin_path)) != meta.get("tokenizer_pin"):
        reasons.append("tokenizer_pin")
    if meta.get("num_shards") != num_shards:
        reasons.append("num_shards")

    # 2. 分片覆盖
    plan = meta["shard_plan"]
    dones = []
    for i in range(num_shards):
        p = data_root / "shards" / f"shard_{i}.done.json"
        if p.exists():
            dones.append(json.loads(p.read_text()))
    n_done = len(dones)
    if n_done != num_shards:
        reasons.append("shards_missing")
    for d in dones:
        p = plan[d["shard"]]
        if (d["row_start"], d["row_end"]) != (p["row_start"], p["row_end"]):
            reasons.append(f"shard{d['shard']}_range")
    ranges = sorted((d["row_start"], d["row_end"]) for d in dones)
    gaps = overlaps = 0
    cursor = 0
    for a, b in ranges:
        if a > cursor:
            gaps += 1
        elif a < cursor:
            overlaps += 1
        cursor = max(cursor, b)
    if cursor < rows:
        gaps += 1
    if gaps or overlaps:
        reasons.append("coverage")

    # 3. 全表扫描
    mm = ext.open_memmap(data_root / ext.MEMMAP_NAME, rows, mode="r")
    nonfinite, zero_rows = scan(mm)
    if nonfinite:
        reasons.append("nonfinite")
    if zero_rows:
        reasons.append("zero_rows")

    # 4. 独立重算（逐位）
    targets = recompute_targets(episodes, plan)
    ok_n = 0
    if encode_fn_factory is None:
        encode_fn_factory = lambda: ext.make_encode_fn(meta["params_path"])  # noqa: E731
    encode_fn = encode_fn_factory()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for g, t, label in targets:
            e = episodes[g]
            chunk_start = (t // ext.BATCH_SIZE) * ext.BATCH_SIZE
            with h5py.File(episode_path(data_root / "episodes", g), "r") as f:
                tokens = ext.encode_episode_chunk(encode_fn, f, chunk_start, pool)
            off = int(e["total_sample_offset"]) + chunk_start
            stored = np.asarray(mm[off : off + tokens.shape[0]])
            if np.array_equal(tokens, stored):
                ok_n += 1
            else:
                diff = float(np.max(np.abs(tokens - stored)))
                print(f"[recompute] 不一致 ep={g} t={t} {label} max_abs={diff:.3e}", flush=True)
    if ok_n != len(targets) or not targets:
        reasons.append("recompute")

    ok = not reasons
    line = (
        f"MEMTOKENS={'PASS' if ok else 'FAIL'} rows={rows} views={NUM_VIEWS} dim={TOKEN_DIM} "
        f"nonfinite={nonfinite} zero_rows={zero_rows} shards={n_done}/{num_shards} gaps={gaps} overlaps={overlaps} "
        f"recompute={ok_n}/{len(targets)}"
    )
    if not ok:
        line += f" reasons={','.join(reasons)}"
    if ok and write_done:
        (data_root / ext.DONE_NAME).write_text(
            json.dumps(
                {
                    "verdict": line,
                    "meta_sha256": ext.sha256_file(data_root / ext.META_NAME),
                    "tokenizer_pin": meta["tokenizer_pin"],
                    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    return ok, line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=ext.DEFAULT_DATA_ROOT)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--pin", default=str(HERE / "tokenizer_pin.json"))
    args = ap.parse_args()
    ok, line = check(args.data_root, args.num_shards, args.pin)
    print(line, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
