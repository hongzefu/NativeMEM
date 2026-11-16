"""S1 训练侧 CPU 单测（计划 C1、G5、G6a、J#4／#9／#11／#12／#19）。

运行（worktree 内）：
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= UV_PROJECT_ENVIRONMENT=/scratch/hongze/NativeMEM/.venv \
    PYTHONPATH=$PWD/src uv run --no-sync python -m pytest scripts/robomme/test_robomme_s1.py -q -s
全部用合成 H5 / 合成 memmap；只有权重加载一项读真实 Stage 1 存档（只读）。
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENPI_DATA_HOME", "/scratch/hongze/.cache/openpi")

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import openpi  # noqa: E402
import openpi.shared.normalize as _normalize  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
from openpi.policies import robomme_policy  # noqa: E402
from openpi.robomme import dataset as rds  # noqa: E402
from openpi.robomme import mem_encoder  # noqa: E402

STAGE1_PARAMS = "/scratch/hongze/NativeMEM/checkpoints/mem_tokenizer_pretrain/pretrain/49999/params"
CONFIG_NAME = "nativemem_pi05_robomme4task"


def test_openpi_points_to_this_checkout():
    print(f"openpi.__file__={openpi.__file__}")
    assert pathlib.Path(openpi.__file__).resolve().is_relative_to(HERE.parents[1].resolve())


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def _jpeg(color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((224, 224, 3), np.uint8)
    img[:] = color
    img[:8, :8] = 255 - np.array(color, np.uint8)  # 让两路可区分
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.reshape(-1)


def make_synthetic(root: pathlib.Path, lengths: list[tuple[int, int]], seed: int = 0) -> dict:
    """lengths: [(num_timesteps, exec_start_idx), ...]；返回路径与真值。"""
    rng = np.random.default_rng(seed)
    ep_dir = root / "episodes"
    ep_dir.mkdir(parents=True)
    episodes, total_off, exec_off = [], 0, 0
    truth = {}
    for g, (n, s) in enumerate(lengths):
        state = rng.normal(size=(n, 8)).astype(np.float32)
        state[:, 7] = rng.uniform(0, 1, size=n)
        action = rng.normal(size=(n, 8)).astype(np.float32)
        is_demo = np.arange(n) < s
        vlen = h5py.vlen_dtype(np.uint8)
        with h5py.File(ep_dir / f"ep_{g:04d}.h5", "w") as f:
            fj = f.create_dataset("front_jpeg", (n,), dtype=vlen)
            wj = f.create_dataset("wrist_jpeg", (n,), dtype=vlen)
            for t in range(n):
                fj[t] = _jpeg((10 + g, (t * 3) % 250, 40))
                wj[t] = _jpeg((200, (t * 5) % 250, 30 + g))
            f["state"] = state
            f["action"] = action
            f["is_demo"] = is_demo
            for k, v in dict(
                global_episode_idx=g, task=f"Task{g % 2}", difficulty="easy", episode=g, seed=g,
                member=f"m/{g}.h5", h5_sha256=f"{g:064x}", num_timesteps=n, exec_start_idx=s,
                task_goal=f"do task {g % 2}",
            ).items():
                f.attrs[k] = v
        episodes.append(
            dict(global_episode_idx=g, h5_file=f"record_dataset_Task{g % 2}.h5", raw_ep_idx=g, num_timesteps=n,
                 exec_start_idx=s, exec_samples=n - s, exec_sample_offset=exec_off, total_sample_offset=total_off,
                 shard_idx=0)
        )
        truth[g] = dict(state=state, action=action)
        total_off += n
        exec_off += n - s
    manifest = root / "episode_manifest.json"
    manifest.write_text(json.dumps({"version": 1, "episodes": episodes}))
    mem = np.memmap(root / "mem_tokens.f32.bin", dtype=np.float32, mode="w+", shape=(total_off, 2, 1152))
    rows = np.arange(total_off, dtype=np.float32)
    mem[:, 0, 0] = rows
    mem[:, 1, 0] = rows
    mem[:, 0, 1] = 0.0
    mem[:, 1, 1] = 1.0
    mem[:, :, 2:] = 0.5
    mem.flush()
    del mem
    return dict(root=root, manifest=manifest, memmap=root / "mem_tokens.f32.bin", episodes=episodes, truth=truth,
                total_rows=total_off, exec_total=exec_off)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    # 3 集：执行帧共 256 个（便于 batch 32 整除）；第 1 集执行帧尾部 < 50 覆盖动作块夹取。
    return make_synthetic(tmp_path_factory.mktemp("synth"), [(200, 90), (130, 60), (176, 100)])


def make_dataset(s, **kw):
    return rds.RoboMMEMemDataset(s["root"] / "episodes", s["manifest"], s["memmap"], **kw)


# ---------------------------------------------------------------------------
# 1. 记忆帧号精确数组（J#4）
# ---------------------------------------------------------------------------
def test_memory_frame_indices_exact():
    assert rds.memory_frame_indices(40, 9) == [0, 4, 13, 22, 31, 40]
    assert rds.memory_frame_indices(25, 12) == [0, 1, 13, 25]
    assert rds.memory_frame_indices(0, 8) == [0]
    assert rds.memory_frame_indices(16, 8) == [0, 8, 16]


# ---------------------------------------------------------------------------
# 2. 动作块越界夹取
# ---------------------------------------------------------------------------
def test_action_chunk_clamp(synth):
    act = synth["truth"][1]["action"]  # n=130
    chunk = rds.action_chunk(act, 100, 50)
    assert chunk.shape == (50, 8)
    np.testing.assert_array_equal(chunk[:30], act[100:130])
    np.testing.assert_array_equal(chunk[30:], np.repeat(act[129:130], 20, axis=0))
    ds = make_dataset(synth)
    ep, t = 1, 100
    idx = synth["episodes"][1]["exec_sample_offset"] + (t - 60)
    assert ds.locate(idx) == (ep, t)
    np.testing.assert_array_equal(ds[idx]["actions"], chunk)
    # 末帧：50 步全部是末帧动作
    last_idx = synth["episodes"][1]["exec_sample_offset"] + (129 - 60)
    np.testing.assert_array_equal(ds[last_idx]["actions"], np.repeat(act[129:130], 50, axis=0))


# ---------------------------------------------------------------------------
# 3. 跨 epoch 回绕（G6a）：不报错、每个 epoch 不重复不遗漏
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class KeepIndex(_transforms.DataTransformFn):
    def __call__(self, data):
        return {"sample_index": data["sample_index"], "mem_interval": data["mem_interval"]}


def test_epoch_wraparound(synth):
    from openpi.training.data_loader import TorchDataLoader
    from openpi.training.data_loader import TransformedDataset

    ds = make_dataset(synth)
    assert len(ds) == synth["exec_total"] == 256
    loader = TorchDataLoader(
        TransformedDataset(ds, [KeepIndex()]), local_batch_size=32, shuffle=True, num_batches=20, num_workers=2, seed=42
    )
    batches = [(np.asarray(b["sample_index"]), np.asarray(b["mem_interval"])) for b in loader]
    assert len(batches) == 20
    per_epoch = len(ds) // 32
    for e in range(2):
        idx = np.concatenate([b[0] for b in batches[e * per_epoch : (e + 1) * per_epoch]])
        assert len(idx) == len(np.unique(idx)) == len(ds), f"epoch {e} 有重复或遗漏"
        assert set(idx.tolist()) == set(range(len(ds)))
    intervals = np.concatenate([b[1] for b in batches])
    assert intervals.min() >= 8 and intervals.max() <= 16
    first, second = (np.concatenate([b[0] for b in batches[i * per_epoch : (i + 1) * per_epoch]]) for i in range(2))
    assert not np.array_equal(first, second), "两个 epoch 顺序相同，shuffle 未生效"


def test_worker_seeding_differs():
    """各 worker 的间隔随机流按 get_worker_info().seed 播种，不再相同（计划 V7）。"""
    import random

    import torch

    class FakeInfo:
        def __init__(self, i, seed):
            self.id, self.seed = i, seed

    ds = rds.RoboMMEMemDataset.__new__(rds.RoboMMEMemDataset)
    ds._rng, ds._rng_owner, ds._seed = None, None, 0
    seqs = []
    orig = torch.utils.data.get_worker_info
    try:
        for i, seed in ((0, 1001), (1, 1002)):
            torch.utils.data.get_worker_info = lambda i=i, seed=seed: FakeInfo(i, seed)
            ds._rng = None
            rng = ds._get_rng()
            seqs.append([rng.randint(8, 16) for _ in range(32)])
            assert seqs[-1] == [random.Random(seed).randint(8, 16) for _ in range(1)] + seqs[-1][1:]
    finally:
        torch.utils.data.get_worker_info = orig
    assert seqs[0] != seqs[1]


# ---------------------------------------------------------------------------
# 4. 合成 H5 + 合成 memmap：键与形状
# ---------------------------------------------------------------------------
def test_dataset_keys_shapes(synth):
    ds = make_dataset(synth)
    idx = synth["episodes"][2]["exec_sample_offset"] + 37
    ep, t = ds.locate(idx)
    assert (ep, t) == (2, 137)
    item = ds[idx]
    assert item["observation/image"].shape == (224, 224, 3) and item["observation/image"].dtype == np.uint8
    assert item["observation/wrist_image"].shape == (224, 224, 3)
    assert not np.array_equal(item["observation/image"], item["observation/wrist_image"])
    np.testing.assert_array_equal(item["observation/state"], synth["truth"][2]["state"][137])
    assert item["actions"].shape == (50, 8) and item["actions"].dtype == np.float32
    np.testing.assert_array_equal(item["actions"], rds.action_chunk(synth["truth"][2]["action"], 137, 50))
    assert item["prompt"] == "do task 0"
    mem = item["memory"]
    frames = rds.memory_frame_indices(t, int(item["mem_interval"]))
    assert mem.shape == (len(frames), 2, 1152) and mem.dtype == np.float32
    expect_rows = synth["episodes"][2]["total_sample_offset"] + np.asarray(frames)
    np.testing.assert_array_equal(mem[:, 0, 0], expect_rows)  # 行号 = total_sample_offset + t（全帧）
    np.testing.assert_array_equal(mem[:, 1, 1], 1.0)  # 视角顺序 [front, wrist]
    np.testing.assert_array_equal(mem[:, 0, 1], 0.0)
    assert frames[0] == 0 and frames[-1] == t and np.all(np.diff(frames) > 0)
    # 当前帧图像与 offline_frame 解码一致
    with h5py.File(synth["root"] / "episodes" / "ep_0002.h5") as f:
        np.testing.assert_array_equal(item["observation/image"], mem_encoder.offline_frame(f["front_jpeg"][137]))
    # 记忆可含演示帧（第 0 帧在演示段内）
    assert frames[0] < synth["episodes"][2]["exec_start_idx"]


def test_offline_online_frame():
    raw = np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    out = mem_encoder.online_frame(raw)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    jb = _jpeg((1, 2, 3))
    dec = mem_encoder.offline_frame(jb)
    np.testing.assert_array_equal(dec, cv2.imdecode(jb, cv2.IMREAD_COLOR))
    np.testing.assert_array_equal(mem_encoder.offline_frame(jb.tobytes()), dec)


# ---------------------------------------------------------------------------
# 5. 完整输入变换；6. 逆变换数值回环
# ---------------------------------------------------------------------------
def _synthetic_norm_stats(assets_dir: pathlib.Path) -> dict:
    rng = np.random.default_rng(1)
    stats = {}
    for k in ("state", "actions"):
        q01 = rng.uniform(-2.0, -0.5, size=8).astype(np.float32)
        q99 = rng.uniform(0.5, 2.0, size=8).astype(np.float32)
        stats[k] = _normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=q01, q99=q99)
    _normalize.save(assets_dir / "robomme4task", stats)
    return stats


def _data_config(synth, tmp_path):
    from openpi.training import config_nativemem as cn

    cfg = cn.get_config(CONFIG_NAME)
    _synthetic_norm_stats(tmp_path / "assets")
    factory = dataclasses.replace(
        cfg.data,
        data_root=str(synth["root"]),
        manifest_path=str(synth["manifest"]),
        memmap_path=str(synth["memmap"]),
        assets=cn.AssetsConfig(assets_dir=str(tmp_path / "assets"), asset_id="robomme4task"),
    )
    return cfg, factory.create(cfg.assets_dirs, cfg.model)


def test_config_matches_original():
    from openpi.training import config_nativemem as cn
    from openpi.training import optimizer as opt
    from openpi.training import weight_loaders as wl

    cfg = cn.get_config(CONFIG_NAME)
    base = cn.get_config("nativemem_pi05")
    assert cfg.batch_size == 128 and cfg.fsdp_devices == 8
    for f in ("num_train_steps", "lr_schedule", "optimizer", "ema_decay", "seed", "save_interval", "keep_period",
              "num_workers", "log_interval", "freeze_filter"):
        assert getattr(cfg, f) == getattr(base, f), f
    assert cfg.lr_schedule == opt.CosineDecaySchedule(warmup_steps=1000, peak_lr=2e-5, decay_steps=20000, decay_lr=1e-6)
    assert dataclasses.replace(cfg.model, memory_tokenizer_max_len=base.model.memory_tokenizer_max_len) == base.model
    assert cfg.model.action_horizon == 50 and cfg.model.memory_tokenizer_max_len == 640
    assert isinstance(cfg.weight_loader, wl.NativeMEMWeightLoader)
    assert cfg.data.assets.asset_id == "robomme4task"


def test_full_input_transform(synth, tmp_path):
    from openpi.models import model as _model
    from openpi.training.data_loader import transform_dataset

    cfg, data_cfg = _data_config(synth, tmp_path)
    assert data_cfg.dataset_kind == "robomme"
    from openpi.training.data_loader_mem import create_mem_torch_dataset

    raw = create_mem_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model, seed=0)
    assert isinstance(raw, rds.RoboMMEMemDataset)
    ds = transform_dataset(raw, data_cfg)
    item = ds[17]
    assert set(item["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for v in item["image"].values():
        assert v.shape == (224, 224, 3)
    assert not np.any(item["image"]["right_wrist_0_rgb"])
    assert bool(item["image_mask"]["base_0_rgb"]) and bool(item["image_mask"]["left_wrist_0_rgb"])
    assert not bool(item["image_mask"]["right_wrist_0_rgb"])
    assert item["state"].shape == (32,) and item["actions"].shape == (50, 32)
    assert np.all(item["actions"][:, 8:] == 0) and np.all(item["state"][8:] == 0)
    assert item["tokenized_prompt"].shape == (cfg.model.max_token_len,)
    assert item["memory"].shape == (640, 1152) and item["memory_mask"].shape == (640,)
    n_mem = int(item["memory_mask"].sum())
    assert n_mem % 2 == 0 and n_mem > 0 and not np.any(item["memory"][n_mem:])
    batch = {k: (v[None] if not isinstance(v, dict) else {kk: vv[None] for kk, vv in v.items()}) for k, v in item.items()}
    obs = _model.Observation.from_dict(batch)
    spec_obs, spec_act = cfg.model.inputs_spec(batch_size=1)
    assert obs.memory.shape == spec_obs.memory.shape and obs.state.shape == spec_obs.state.shape
    assert batch["actions"].shape == spec_act.shape


def test_inverse_roundtrip(synth, tmp_path):
    cfg, data_cfg = _data_config(synth, tmp_path)
    norm_stats = data_cfg.norm_stats
    rng = np.random.default_rng(3)
    state = rng.uniform(-1, 1, size=8).astype(np.float32)
    state[:7] += 0.7  # 非零 state
    actions = rng.uniform(-1, 1, size=(50, 8)).astype(np.float32)
    sample = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": state.copy(),
        "actions": actions.copy(),
        "prompt": "x",
    }
    fwd = _transforms.compose([
        *data_cfg.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=True),
        _transforms.PadStatesAndActions(cfg.model.action_dim),
    ])
    out = fwd(sample)
    # 正变换确实做了 delta：反归一化后的 delta 关节 = 动作 − state
    inv = _transforms.compose([
        *data_cfg.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=True),
        *data_cfg.data_transforms.outputs,
    ])
    back = inv({"state": out["state"].copy(), "actions": out["actions"].copy()})
    assert back["actions"].shape == (50, 8)
    assert np.max(np.abs(back["actions"] - actions)) <= 1e-5
    # 输出链顺序：Unnormalize → AbsoluteActions → RoboMMEOutputs
    names = [type(t).__name__ for t in data_cfg.data_transforms.outputs]
    assert names == ["AbsoluteActions", "RoboMMEOutputs"]
    assert data_cfg.data_transforms.outputs[0].mask == _transforms.make_bool_mask(7, -1)


# ---------------------------------------------------------------------------
# 7. 归一化统计
# ---------------------------------------------------------------------------
def test_norm_stats_semantics(synth, tmp_path):
    import compute_norm_stats_robomme as cns

    st = synth["truth"][1]["state"]
    ac = synth["truth"][1]["action"]
    st0, ac0 = st.copy(), ac.copy()
    s1, a1 = cns.episode_delta_samples(st, ac, 60, 50)
    s2, a2 = cns.episode_delta_samples(st, ac, 60, 50)
    np.testing.assert_array_equal(st, st0)  # 底层数据不变
    np.testing.assert_array_equal(ac, ac0)
    assert len(s1) == 130 - 60
    for x, y in zip(a1, a2, strict=True):
        np.testing.assert_array_equal(x, y)  # 重复读取结果相同
    t = 100
    k = t - 60
    np.testing.assert_allclose(a1[k][:30, :7], ac[100:130, :7] - st[t, :7], atol=0)  # 关节减锚点 state
    np.testing.assert_array_equal(a1[k][:, 7], np.concatenate([ac[100:130, 7], np.repeat(ac[129, 7], 20)]))  # 夹爪绝对
    np.testing.assert_allclose(a1[k][30:, :7], np.repeat(ac[129:130, :7] - st[t, :7], 20, axis=0))  # 尾部重复末帧
    np.testing.assert_array_equal(s1[k], st[t])
    out = tmp_path / "ns"
    _, n_states, n_actions = cns.compute(str(synth["root"]), str(synth["manifest"]), str(out), workers=2)
    assert n_states == synth["exec_total"] and n_actions == 50 * synth["exec_total"]
    loaded = _normalize.load(out)
    assert loaded["state"].q01.shape == (8,) and loaded["actions"].q99.shape == (8,)


# ---------------------------------------------------------------------------
# 8. 权重加载（G5）
# ---------------------------------------------------------------------------
def test_weights_load():
    import flax.traverse_util as tu
    import jax

    from openpi.models import model as _model
    from openpi.shared import array_typing as at
    from openpi.training import config_nativemem as cn
    from openpi.training import weight_loaders as wl

    if not pathlib.Path(STAGE1_PARAMS).exists():
        pytest.skip("Stage 1 权重不在本机")
    cfg = cn.get_config(CONFIG_NAME)
    t0 = time.time()
    import flax.nnx as nnx

    # 与 scripts/train_nativemem.py::init_train_state 相同：eval_shape(init) 得到参数形状树。
    params_shape = jax.eval_shape(lambda r: nnx.state(cfg.model.create(r)), jax.random.key(0)).to_pure_dict()
    loader = wl.NativeMEMWeightLoader(STAGE1_PARAMS)
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    flat = tu.flatten_dict(loaded, sep="/")
    random_init = [k for k, v in flat.items() if isinstance(v, jax.ShapeDtypeStruct)]
    ckpt_flat = tu.flatten_dict(_model.restore_params(STAGE1_PARAMS, restore_type=np.ndarray), sep="/")
    ref_keys = set(tu.flatten_dict(params_shape, sep="/"))

    def used(k):
        return k in ref_keys or (k.endswith("/value") and k[:-6] in ref_keys)

    unused = sorted({k.split("/")[0] for k in ckpt_flat if not used(k)})
    elapsed = time.time() - t0
    print(f"[weights] ref={len(ref_keys)} ckpt={len(ckpt_flat)} random_init={len(random_init)} unused_top={unused} {elapsed:.0f}s")
    assert random_init == [], random_init[:5]
    assert unused == ["encoder"], unused
    print("WEIGHTS_LOAD=PASS loader=NativeMEMWeightLoader src=stage1-49999 random_init=0 unused=encoder")


def test_mem_encoder_loads_and_encodes():
    if not pathlib.Path(STAGE1_PARAMS).exists():
        pytest.skip("Stage 1 权重不在本机")
    enc = mem_encoder.load_encoder(STAGE1_PARAMS)
    frames = np.stack([mem_encoder.offline_frame(_jpeg((50, 100, 150))), mem_encoder.online_frame(np.zeros((256, 256, 3), np.uint8))])
    tok = mem_encoder.encode_frames(enc, frames)
    assert tok.shape == (2, 1152) and tok.dtype == np.float32 and np.all(np.isfinite(tok))
    assert not np.allclose(tok[0], tok[1])
    one = mem_encoder.encode_frames(enc, frames[:1])
    print(f"[encoder] batch1_vs_batch2 max_abs={np.max(np.abs(one[0] - tok[0])):.3e} norm={np.linalg.norm(tok[0]):.3f}")
    assert np.max(np.abs(one[0] - tok[0])) <= 1e-4


# ---------------------------------------------------------------------------
# 9. check_finite 遇 NaN 必须 FAIL（J#9）
# ---------------------------------------------------------------------------
def test_check_finite_detects_nan():
    import check_finite as cf

    ok, msg = cf.selftest()
    assert ok, msg
    assert not cf.check_log_text("Step 3: loss=nan, grad_norm=1.0, param_norm=2.0")[0]
    assert not cf.check_tree_finite({"w": np.array([1.0, np.inf], np.float32)})[0]
    bad = np.zeros((50, 8), np.float32)
    bad[0, 0] = np.nan
    assert not cf.check_actions(bad)[0]


# ---------------------------------------------------------------------------
# 抽 token 与 G7 检查在合成数据上的端到端（用假编码函数，不加载权重）
# ---------------------------------------------------------------------------
def _fake_encode(frames):
    f = np.asarray(frames, np.float32)
    out = np.zeros((f.shape[0], 1152), np.float32)
    out[:, 0] = f.mean(axis=(1, 2, 3)) + 1.0
    out[:, 1] = f[:, :8, :8].mean(axis=(1, 2, 3)) + 1.0
    return out


def test_extract_and_check_memtokens(tmp_path):
    import check_memtokens as cm
    import extract_mem_tokens as ext

    s = make_synthetic(tmp_path / "d", [(70, 30), (130, 64), (65, 20), (90, 45)])
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"revision": "a" * 40, "manifest_sha256": "b" * 64, "repo_id": "x/y"}))
    (s["root"] / ext.MEMMAP_NAME).unlink()  # 合成 memmap 由 init 重新建
    args = type("A", (), dict(data_root=str(s["root"]), manifest=str(s["manifest"]), params_path="unused",
                              pin=str(pin), num_shards=3))()
    ext.cmd_init(args)
    with pytest.raises(SystemExit):
        ext.cmd_init(args)  # 拒绝覆盖
    # 只跑 2 片时检查必须 FAIL（缺口 + 全零行）
    for i in range(2):
        ext.run_shard(s["root"], i, _fake_encode)
    ok, line = cm.check(s["root"], 3, pin, lambda: _fake_encode, write_done=False)
    print(line)
    assert not ok and "zero_rows=0" not in line and "shards=2/3" in line
    ext.run_shard(s["root"], 2, _fake_encode)
    ok, line = cm.check(s["root"], 3, pin, lambda: _fake_encode)
    print(line)
    assert ok, line
    assert (s["root"] / ext.DONE_NAME).exists()
    # 行号规则：row = total_sample_offset + t
    mm = ext.open_memmap(s["root"] / ext.MEMMAP_NAME, s["total_rows"])
    e = s["episodes"][1]
    with h5py.File(s["root"] / "episodes" / "ep_0001.h5") as f:
        front = mem_encoder.offline_frame(f["front_jpeg"][77])
        wrist = mem_encoder.offline_frame(f["wrist_jpeg"][77])
    np.testing.assert_allclose(mm[e["total_sample_offset"] + 77, 0, :2], _fake_encode(front[None])[0, :2])
    np.testing.assert_allclose(mm[e["total_sample_offset"] + 77, 1, :2], _fake_encode(wrist[None])[0, :2])
    # 篡改一行：重算或全零检查必须 FAIL
    del mm
    mmw = np.memmap(s["root"] / ext.MEMMAP_NAME, dtype=np.float32, mode="r+").reshape(-1, 2, 1152)
    mmw[e["total_sample_offset"] + e["exec_start_idx"], 1, :] = 0.0
    mmw.flush()
    del mmw
    ok, line = cm.check(s["root"], 3, pin, lambda: _fake_encode, write_done=False)
    print(line)
    assert not ok and "zero_rows=1" in line


def test_mem_data_loader_end_to_end(synth, tmp_path):
    """训练入口 create_mem_torch_data_loader 走 RoboMME 分支，产出 (Observation, Actions) 批次。"""
    from openpi.training.data_loader_mem import create_mem_torch_data_loader

    cfg, data_cfg = _data_config(synth, tmp_path)
    loader = create_mem_torch_data_loader(
        data_cfg, cfg.model, cfg.model.action_horizon, 4, shuffle=True, num_batches=2, num_workers=0, seed=42
    )
    batches = list(loader)
    assert len(batches) == 2
    obs, act = batches[0]
    assert act.shape == (4, 50, 32)
    assert obs.state.shape == (4, 32)
    assert obs.memory.shape == (4, 640, 1152) and obs.memory_mask.shape == (4, 640)
    assert obs.images["base_0_rgb"].shape == (4, 224, 224, 3)
    assert obs.tokenized_prompt.shape == (4, cfg.model.max_token_len)
    assert not bool(np.asarray(obs.image_masks["right_wrist_0_rgb"]).any())
