"""RoboMME 4task 的 NativeMEM Stage 2 训练数据集（取样语义照原版 MemLeRobotDataset，只换容器，计划 X8）。

数据来源（S0 build_data_v2 产出，每集一个 H5）：
  <episodes_dir>/ep_<global_episode_idx:04d>.h5
    front_jpeg、wrist_jpeg：vlen uint8 [T]（resize_with_pad_cv2 224 + JPEG q95，RGB 原样）
    state：f32 [T,8]；action：f32 [T,8] = joint_action[t]（不移位）；is_demo：bool [T]
    attrs：global_episode_idx、task、num_timesteps、exec_start_idx、task_goal（小写）……
记忆 token（S1 extract_mem_tokens 产出）：
  mem_tokens.f32.bin，np.memmap f32 [N_total, 2, 1152]，视角 [front, wrist]，
  行号 row = episode_manifest.episodes[e].total_sample_offset + t（t 为全帧号，含演示帧）。

样本（计划 X3）：只用执行帧 t ∈ [exec_start_idx, num_timesteps)，共 605,611 个；按帧均匀（DataLoader 原版 shuffle）。
每个样本：
  - 当前帧两路 JPEG 解码（offline_frame）、state[t]、prompt=task_goal；
  - 动作块 action[t..t+49]，越界夹到 episode 末帧（原版 LeRobot delta_timestamps 行为）；
  - 记忆帧号：每样本抽一次 interval ∈ [8,16]，range(t, -1, -interval)，末尾不是 0 就补 0，反转为升序
    （原版 MemLeRobotDataset._get_memory_indices，ep_start 对应本集第 0 帧，含演示帧）；
    查 memmap 得 memory f32 [T_mem, 2, 1152]。
  - 随机流按 torch get_worker_info().seed 播种（原版各 worker 用同一个 random.Random(seed)，间隔序列相同，计划 V7 修正）。
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
from typing import Any

import h5py
import numpy as np
import torch

from openpi.robomme.mem_encoder import TOKEN_DIM
from openpi.robomme.mem_encoder import offline_frame

NUM_VIEWS = 2
VIEW_ORDER = ("front", "wrist")


def memory_frame_indices(t: int, interval: int, ep_start: int = 0) -> list[int]:
    """原版 MemLeRobotDataset._get_memory_indices 的纯函数版本（帧号相对本集第 0 帧）。"""
    if interval <= 0:
        raise ValueError(f"interval 必须为正，得到 {interval}")
    indices = list(range(t, ep_start - 1, -interval))
    if indices[-1] != ep_start:
        indices.append(ep_start)
    indices.reverse()
    return indices


def action_chunk(action: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """action[t..t+horizon-1]，越界部分重复本集末帧动作（LeRobot 夹到 episode 末帧）。"""
    num = action.shape[0]
    idx = np.minimum(np.arange(t, t + horizon), num - 1)
    return np.asarray(action[idx], dtype=np.float32)


def load_manifest(manifest_path: str | pathlib.Path) -> list[dict[str, Any]]:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"] if isinstance(data, dict) else data
    episodes = sorted(episodes, key=lambda e: e["global_episode_idx"])
    for i, e in enumerate(episodes):
        if e["global_episode_idx"] != i:
            raise ValueError(f"episode_manifest 的 global_episode_idx 不连续：位置 {i} 是 {e['global_episode_idx']}")
    return episodes


def episode_path(episodes_dir: str | pathlib.Path, global_idx: int) -> pathlib.Path:
    return pathlib.Path(episodes_dir) / f"ep_{global_idx:04d}.h5"


class RoboMMEMemDataset(torch.utils.data.Dataset):
    """按执行帧索引的 map-style 数据集；输出键与推理输入键相同（见 openpi.policies.robomme_policy）。"""

    def __init__(
        self,
        episodes_dir: str | pathlib.Path,
        manifest_path: str | pathlib.Path,
        memmap_path: str | pathlib.Path,
        *,
        action_horizon: int = 50,
        intervals: tuple[int, int] = (8, 16),
        seed: int = 0,
        max_open_files: int = 64,
    ):
        self._episodes_dir = pathlib.Path(episodes_dir)
        self._memmap_path = pathlib.Path(memmap_path)
        self._action_horizon = action_horizon
        self._intervals = (int(intervals[0]), int(intervals[1]))
        self._seed = seed
        self._max_open_files = max_open_files

        episodes = load_manifest(manifest_path)
        self._num_timesteps = np.array([e["num_timesteps"] for e in episodes], dtype=np.int64)
        self._exec_start = np.array([e["exec_start_idx"] for e in episodes], dtype=np.int64)
        self._total_offset = np.array([e["total_sample_offset"] for e in episodes], dtype=np.int64)
        exec_counts = self._num_timesteps - self._exec_start
        if np.any(exec_counts <= 0):
            raise ValueError("存在没有执行帧的 episode")
        # 执行帧样本偏移按 global 顺序累加；与 manifest 的 exec_sample_offset 一致性在此核对。
        self._exec_offset = np.concatenate([[0], np.cumsum(exec_counts)[:-1]]).astype(np.int64)
        if "exec_sample_offset" in episodes[0]:
            given = np.array([e["exec_sample_offset"] for e in episodes], dtype=np.int64)
            if not np.array_equal(given, self._exec_offset):
                raise ValueError("episode_manifest 的 exec_sample_offset 与执行帧计数累加不一致")
        expect_total = np.concatenate([[0], np.cumsum(self._num_timesteps)[:-1]])
        if not np.array_equal(expect_total, self._total_offset):
            raise ValueError("episode_manifest 的 total_sample_offset 与帧数累加不一致")
        self._len = int(exec_counts.sum())
        self._total_rows = int(self._num_timesteps.sum())

        # 以下在各进程内惰性创建（DataLoader 用 spawn，对象被 pickle 到 worker）。
        self._rng: random.Random | None = None
        self._rng_owner: Any = None
        self._mem: np.memmap | None = None
        self._files: collections.OrderedDict[int, tuple[h5py.File, str]] = collections.OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rng"] = None
        state["_rng_owner"] = None
        state["_mem"] = None
        state["_files"] = collections.OrderedDict()
        return state

    def __len__(self) -> int:
        return self._len

    @property
    def total_rows(self) -> int:
        return self._total_rows

    # ---- 惰性资源 ----
    def _get_rng(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        owner = ("worker", info.id, info.seed) if info is not None else ("main",)
        if self._rng is None or self._rng_owner != owner:
            seed = info.seed if info is not None else self._seed
            self._rng = random.Random(seed)
            self._rng_owner = owner
        return self._rng

    def _get_mem(self) -> np.memmap:
        if self._mem is None:
            mem = np.memmap(self._memmap_path, dtype=np.float32, mode="r")
            expect = self._total_rows * NUM_VIEWS * TOKEN_DIM
            if mem.size != expect:
                raise ValueError(f"{self._memmap_path} 元素数 {mem.size} ≠ 期望 {expect}（{self._total_rows}×2×{TOKEN_DIM}）")
            self._mem = mem.reshape(self._total_rows, NUM_VIEWS, TOKEN_DIM)
        return self._mem

    def _get_file(self, ep: int) -> tuple[h5py.File, str]:
        if ep in self._files:
            self._files.move_to_end(ep)
            return self._files[ep]
        f = h5py.File(episode_path(self._episodes_dir, ep), "r")
        n = int(f.attrs["num_timesteps"])
        if n != int(self._num_timesteps[ep]) or int(f.attrs["exec_start_idx"]) != int(self._exec_start[ep]):
            f.close()
            raise ValueError(f"ep {ep} 的 H5 attrs 与 episode_manifest 不一致")
        if int(f.attrs["global_episode_idx"]) != ep:
            f.close()
            raise ValueError(f"ep {ep} 的 H5 global_episode_idx 不符")
        goal = f.attrs["task_goal"]
        goal = goal.decode() if isinstance(goal, bytes) else str(goal)
        self._files[ep] = (f, goal)
        while len(self._files) > self._max_open_files:
            _, (old, _) = self._files.popitem(last=False)
            old.close()
        return self._files[ep]

    # ---- 取样 ----
    def locate(self, idx: int) -> tuple[int, int]:
        """执行帧样本号 → (global_episode_idx, 本集帧号 t)。"""
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        ep = int(np.searchsorted(self._exec_offset, idx, side="right") - 1)
        t = int(self._exec_start[ep] + (idx - self._exec_offset[ep]))
        return ep, t

    def memory_rows(self, ep: int, frame_indices: list[int]) -> np.ndarray:
        return self._total_offset[ep] + np.asarray(frame_indices, dtype=np.int64)

    def __getitem__(self, idx) -> dict:
        idx = int(idx)
        ep, t = self.locate(idx)
        f, goal = self._get_file(ep)

        front = offline_frame(f["front_jpeg"][t])
        wrist = offline_frame(f["wrist_jpeg"][t])
        state = np.asarray(f["state"][t], dtype=np.float32)
        n = int(self._num_timesteps[ep])
        stop = min(t + self._action_horizon, n)
        window = np.asarray(f["action"][t:stop], dtype=np.float32)
        if window.shape[0] < self._action_horizon:
            pad = np.repeat(window[-1:], self._action_horizon - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)

        interval = self._get_rng().randint(self._intervals[0], self._intervals[1])
        frames = memory_frame_indices(t, interval)
        rows = self.memory_rows(ep, frames)
        memory = np.array(self._get_mem()[rows], dtype=np.float32)  # 拷贝出 memmap

        return {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "actions": window,
            "prompt": goal,
            "memory": memory,
            # 以下是调试元数据，RoboMMEMemInputs 不透传，不进模型。
            "sample_index": np.int64(idx),
            "episode_index": np.int64(ep),
            "frame_index": np.int64(t),
            "mem_interval": np.int64(interval),
        }
