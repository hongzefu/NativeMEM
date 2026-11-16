"""RoboMME 4task 归一化统计（计划 C1、X6、J#11；仿官方 scripts/prepare_mem_tokenizer_h5.py::accumulate_episode_stats）。

口径：
- 只遍历执行帧 t ∈ [exec_start_idx, num_timesteps)（与训练样本完全相同，共 605,611 个）；
- 每帧取 state[t] 与未来 50 步动作窗 action[t..t+49]，窗口不足时重复本集末帧动作；
- 依次过 RoboMMEInputs → DeltaActions(make_bool_mask(7,-1))：50 步动作全部减当前锚点 state 的前 7 维，
  夹爪（第 8 维）保持绝对；变换前先 copy()，不原地改动 H5 读出的数组；
- normalize.RunningStats 逐帧 update（state 一行、动作窗 50 行，与原版顺序和粒度相同），
  按 global_episode_idx 升序累计；
- 存 <out_dir>/norm_stats.json（mean/std/q01/q99，8 维），训练用分位数归一化。
读取 H5 用进程池并行，统计累计在主进程按集序串行（RunningStats 直方图与更新顺序有关，保持原版逐帧粒度）。

末行：NORM_STATS=PASS states=605611 action_vectors=30280550 sha256=<norm_stats.json sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import openpi.shared.normalize as normalize  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
from openpi.policies import robomme_policy  # noqa: E402
from openpi.robomme.dataset import episode_path  # noqa: E402
from openpi.robomme.dataset import load_manifest  # noqa: E402

DEFAULT_DATA_ROOT = "/scratch/hongze/NativeMEM/data/robomme4task-v2"
DEFAULT_MANIFEST = (
    "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/"
    "4task-v2-1600ep-604f16da/meta/episode_manifest.json"
)


def episode_delta_samples(
    state: np.ndarray, action: np.ndarray, exec_start: int, action_horizon: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """对一集的每个执行帧给出 (state, 50 步 delta 动作窗)，与训练变换链 RoboMMEInputs→DeltaActions 相同。

    不修改传入的 state／action（每帧都先 copy）。
    """
    inputs = robomme_policy.RoboMMEInputs()
    delta = _transforms.DeltaActions(robomme_policy.DELTA_ACTION_MASK)
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)  # 图像不影响统计
    n = len(state)
    states, actions = [], []
    for t in range(exec_start, n):
        window = action[t : min(t + action_horizon, n)]
        if window.shape[0] < action_horizon:
            pad = np.repeat(window[-1:], action_horizon - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        sample = {
            "observation/image": dummy,
            "observation/wrist_image": dummy,
            "observation/state": state[t].copy(),
            "actions": window.copy(),
        }
        sample = delta(inputs(sample))
        states.append(sample["state"])
        actions.append(sample["actions"])
    return states, actions


def _read_episode(args: tuple[str, int, int, int, int]) -> tuple[int, list[np.ndarray], list[np.ndarray]]:
    episodes_dir, ep, exec_start, num_timesteps, horizon = args
    with h5py.File(episode_path(episodes_dir, ep), "r") as f:
        if int(f.attrs["num_timesteps"]) != num_timesteps or int(f.attrs["exec_start_idx"]) != exec_start:
            raise ValueError(f"ep {ep} 的 H5 attrs 与 episode_manifest 不一致")
        state = np.asarray(f["state"][:], dtype=np.float32)
        action = np.asarray(f["action"][:], dtype=np.float32)
    if state.shape != (num_timesteps, 8) or action.shape != (num_timesteps, 8):
        raise ValueError(f"ep {ep} state/action 形状不符：{state.shape} {action.shape}")
    s, a = episode_delta_samples(state, action, exec_start, horizon)
    return ep, s, a


def compute(
    data_root: str, manifest: str, out_dir: str, *, action_horizon: int = 50, workers: int = 32,
    expect_states: int | None = None,
) -> tuple[dict[str, normalize.NormStats], int, int]:
    episodes = load_manifest(manifest)
    episodes_dir = str(pathlib.Path(data_root) / "episodes")
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    jobs = [(episodes_dir, e["global_episode_idx"], e["exec_start_idx"], e["num_timesteps"], action_horizon) for e in episodes]
    n_states = n_actions = 0
    # spawn：避免 JAX 已初始化线程后 fork 死锁。
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for ep, s_list, a_list in ex.map(_read_episode, jobs, chunksize=4):
            for s, a in zip(s_list, a_list, strict=True):
                stats["state"].update(s[None, :])
                stats["actions"].update(a)
                n_states += 1
                n_actions += a.shape[0]
            if ep % 100 == 0:
                print(f"[norm] ep {ep} states={n_states}", flush=True)
    expect = expect_states if expect_states is not None else sum(e["num_timesteps"] - e["exec_start_idx"] for e in episodes)
    if n_states != expect or n_actions != expect * action_horizon:
        raise SystemExit(f"NORM_STATS=FAIL reason=count states={n_states} expect={expect} action_vectors={n_actions}")
    result = {k: v.get_statistics() for k, v in stats.items()}
    for k, v in result.items():
        for name in ("mean", "std", "q01", "q99"):
            arr = np.asarray(getattr(v, name))
            if arr.shape != (8,) or not np.all(np.isfinite(arr)):
                raise SystemExit(f"NORM_STATS=FAIL reason=bad_{k}_{name} shape={arr.shape}")
    normalize.save(out_dir, result)
    return result, n_states, n_actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "assets" / "robomme4task"))
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()
    _, n_states, n_actions = compute(
        args.data_root, args.manifest, args.out_dir, action_horizon=args.action_horizon, workers=args.workers
    )
    path = pathlib.Path(args.out_dir) / "norm_stats.json"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"NORM_STATS=PASS states={n_states} action_vectors={n_actions} sha256={sha} out={path}")


if __name__ == "__main__":
    main()
