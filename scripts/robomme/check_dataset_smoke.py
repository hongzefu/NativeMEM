"""G6b：真库取样检查（计划 E 表 G6、J#10；第 5 步、G7 之后，只用 CPU）。

对 RoboMMEMemDataset（每集 H5 + 记忆 token memmap）做：
  - len(dataset) == 605611（可用 --expect-len 改，供合成数据单测）；
  - 每个任务（episode_manifest 的 h5_file）各抽若干样本：该任务执行帧均匀取样，外加首集首个执行帧、末集末帧；
  - 键名与形状：observation/image、observation/wrist_image 为 uint8 [224,224,3]，observation/state f32 [8]，
    actions f32 [50,8]，prompt 非空 str，memory f32 [T,2,1152]；
  - 记忆帧号（由样本的 mem_interval 与 frame_index 按原版规则重建）：升序、首个为 0、末个为 t、
    除首段外相邻间隔都等于 interval 且 interval ∈ [8,16]、首段 ∈ (0, interval]（t=0 时只有 [0]）；
  - memory 与独立打开 memmap 读出的 rows = total_sample_offset + 帧号 逐位相等；
  - 当前帧 state 与独立打开 H5 读出的 state[t] 相同；两路图像与 H5 JPEG 解码相同；
    动作块与 H5 action[t..t+49]（越界夹末帧）相同。
memmap 不存在时明确报错（DATASET_SMOKE=FAIL reason=no_memmap）。
末行：DATASET_SMOKE=PASS n=605611 samples=<k>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from openpi.robomme import dataset as rds  # noqa: E402
from openpi.robomme.mem_encoder import offline_frame  # noqa: E402

DEFAULT_DATA_ROOT = "/scratch/hongze/NativeMEM/data/robomme4task-v2"
DEFAULT_MANIFEST = (
    "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/"
    "4task-v2-1600ep-604f16da/meta/episode_manifest.json"
)
EXPECT_LEN = 605611


def pick_samples(episodes: list[dict], per_task: int) -> list[int]:
    """每任务均匀取 per_task 个执行帧样本号，外加首集首个执行帧与末集末帧。"""
    by_task: dict[str, list[dict]] = {}
    for e in episodes:
        by_task.setdefault(e.get("h5_file", "all"), []).append(e)
    picks: set[int] = set()
    exec_off = 0
    offsets = {}
    for e in episodes:  # 执行帧样本号偏移按 global 顺序累加（与数据集一致）
        offsets[e["global_episode_idx"]] = exec_off
        exec_off += e["num_timesteps"] - e["exec_start_idx"]
    for _task, eps in sorted(by_task.items()):
        # 只在本任务各集的执行帧样本号里取（任务在 manifest 里不一定连续）
        idxs = np.concatenate([
            np.arange(offsets[e["global_episode_idx"]],
                      offsets[e["global_episode_idx"]] + e["num_timesteps"] - e["exec_start_idx"])
            for e in eps
        ])
        pos = np.linspace(0, len(idxs) - 1, per_task).round().astype(int)
        picks.update(int(x) for x in idxs[pos])
        picks.update((int(idxs[0]), int(idxs[-1])))
    return sorted(picks)


def check_sample(ds: rds.RoboMMEMemDataset, idx: int, episodes: list[dict], mem: np.ndarray, episodes_dir) -> list[str]:
    errs = []
    try:
        item = ds[idx]
    except Exception as exc:  # 取样本身抛错也记为不符，继续检查其余样本
        return [f"getitem:{type(exc).__name__}:{exc}"]
    ep, t = ds.locate(idx)
    if int(item["episode_index"]) != ep or int(item["frame_index"]) != t:
        errs.append("locate")
    e = episodes[ep]
    if not e["exec_start_idx"] <= t < e["num_timesteps"]:
        errs.append("not_exec_frame")
    for k in ("observation/image", "observation/wrist_image"):
        v = item[k]
        if v.dtype != np.uint8 or v.shape != (224, 224, 3):
            errs.append(f"{k}:{v.dtype}{v.shape}")
    if item["observation/state"].dtype != np.float32 or item["observation/state"].shape != (8,):
        errs.append("state_shape")
    if item["actions"].dtype != np.float32 or item["actions"].shape != (50, 8):
        errs.append("actions_shape")
    if not isinstance(item["prompt"], str) or not item["prompt"]:
        errs.append("prompt")
    m = item["memory"]
    interval = int(item["mem_interval"])
    frames = rds.memory_frame_indices(t, interval)
    if m.dtype != np.float32 or m.shape != (len(frames), 2, 1152):
        errs.append(f"memory_shape:{m.shape}")
    gaps = np.diff(frames)
    if not 8 <= interval <= 16:
        errs.append(f"interval={interval}")
    if frames[0] != 0 or frames[-1] != t or np.any(gaps <= 0):
        errs.append("frames_order")
    if len(frames) > 1 and (np.any(gaps[1:] != interval) or not 0 < gaps[0] <= interval):
        errs.append("frames_gap")
    rows = int(e["total_sample_offset"]) + np.asarray(frames)
    if m.shape == (len(frames), 2, 1152) and not np.array_equal(m, np.asarray(mem[rows])):
        errs.append("memory_rows")
    with h5py.File(rds.episode_path(episodes_dir, ep), "r") as f:
        if not np.array_equal(item["observation/state"], np.asarray(f["state"][t], np.float32)):
            errs.append("state_value")
        if not np.array_equal(item["observation/image"], offline_frame(f["front_jpeg"][t])):
            errs.append("front_pixels")
        if not np.array_equal(item["observation/wrist_image"], offline_frame(f["wrist_jpeg"][t])):
            errs.append("wrist_pixels")
        act = np.asarray(f["action"][:], np.float32)
        if not np.array_equal(item["actions"], rds.action_chunk(act, t, 50)):
            errs.append("actions_value")
        goal = f.attrs["task_goal"]
        if item["prompt"] != (goal.decode() if isinstance(goal, bytes) else str(goal)):
            errs.append("prompt_value")
    return errs


def run_smoke(data_root, manifest, memmap=None, *, expect_len=EXPECT_LEN, per_task=8, seed=0) -> tuple[bool, str]:
    data_root = pathlib.Path(data_root)
    memmap = pathlib.Path(memmap) if memmap else data_root / "mem_tokens.f32.bin"
    if not memmap.exists():
        return False, f"DATASET_SMOKE=FAIL reason=no_memmap path={memmap}（先完成第 4 步抽 token 与 G7）"
    episodes_dir = data_root / "episodes"
    episodes = rds.load_manifest(manifest)
    ds = rds.RoboMMEMemDataset(episodes_dir, manifest, memmap, action_horizon=50, seed=seed)
    n = len(ds)
    if n != expect_len:
        return False, f"DATASET_SMOKE=FAIL reason=len n={n} expect={expect_len}"
    total_rows = sum(e["num_timesteps"] for e in episodes)
    mem = np.memmap(memmap, dtype=np.float32, mode="r").reshape(total_rows, 2, 1152)
    picks = pick_samples(episodes, per_task)
    bad = []
    for idx in picks:
        errs = check_sample(ds, idx, episodes, mem, episodes_dir)
        if errs:
            bad.append((idx, errs))
            print(f"[dataset-smoke] 样本 {idx} 不符：{errs}", flush=True)
    if bad:
        return False, f"DATASET_SMOKE=FAIL n={n} samples={len(picks)} bad={len(bad)}"
    return True, f"DATASET_SMOKE=PASS n={n} samples={len(picks)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--memmap", default=None, help="默认 <data_root>/mem_tokens.f32.bin")
    ap.add_argument("--expect-len", type=int, default=EXPECT_LEN)
    ap.add_argument("--per-task", type=int, default=8)
    args = ap.parse_args()
    ok, line = run_smoke(args.data_root, args.manifest, args.memmap, expect_len=args.expect_len, per_task=args.per_task)
    print(line, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
