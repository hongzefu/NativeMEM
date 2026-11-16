"""serve_nativemem.py 协议单测（CPU、假策略、打桩编码器；1116 计划 C2、J#5 #16 #17）。

覆盖：记忆帧号规则（含演示帧、D 非 12 倍数）；重复 infer 不追加帧；reset 清空；记忆 > 640 行报错；
哨兵端到端：front／wrist 填不同常数像素，经复制版真实 ``pack_buffer`` → policy 仓 openpi_client
``MMEVLAWebsocketClientPolicy``（按源码路径加载）→ websocket → 服务器解包，两路分别还原、帧号正确，
输出 ``CLIENT_WRIST_E2E=PASS``。

运行（S2 验收）：
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= UV_PROJECT_ENVIRONMENT=/scratch/hongze/NativeMEM/.venv \
  PYTHONPATH=$PWD/src uv run --no-sync python -m pytest scripts/robomme/test_serve_protocol.py -q
"""

from __future__ import annotations

import asyncio
import json
import importlib.util
import pathlib
import sys
import threading

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
POLICY_SRC = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
POLICY_WS_CLIENT = POLICY_SRC / "packages/openpi-client/src/openpi_client/websocket_client_policy.py"


def _load(name: str, path: pathlib.Path):
    # policy 仓只读：加载其源码时不写 __pycache__
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


serve = _load("nm_serve_nativemem", HERE / "serve_nativemem.py")
client_utils = _load("nm_client_utils", HERE / "client" / "utils.py")

H = W = 16          # 测试用小图，真实评估为 256×256
TOKEN_DIM = serve.TOKEN_DIM


def stub_online_frame(raw: np.ndarray) -> np.ndarray:
    """打桩 S1 ``online_frame``：只保留像素常数，铺成 224×224。"""
    assert raw.dtype == np.uint8 and raw.ndim == 3
    return np.full((224, 224, 3), raw[0, 0, 0], dtype=np.uint8)


def stub_encode(x: np.ndarray) -> np.ndarray:
    """打桩 S1 ``encode_frames``：必须是 batch=1；token 每维都等于像素常数。"""
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.uint8, x.shape
    return np.full((1, TOKEN_DIM), float(x[0, 0, 0, 0]), dtype=np.float32)


def front_value(f: int) -> int:
    return (2 * f + 1) % 128           # 奇数，落在 [1,127]


def wrist_value(f: int) -> int:
    return 128 + (2 * f) % 128         # 落在 [128,255]，与前视不相交


def frames(start: int, n: int):
    imgs = [np.full((H, W, 3), front_value(f), dtype=np.uint8) for f in range(start, start + n)]
    wrists = [np.full((H, W, 3), wrist_value(f), dtype=np.uint8) for f in range(start, start + n)]
    states = [np.full(8, f, dtype=np.float32) for f in range(start, start + n)]
    return imgs, wrists, states


class FakePolicy:
    metadata = {"fake": True}

    def __init__(self):
        self.calls = []
        self.reseeds = 0

    def infer(self, element: dict) -> dict:
        mem = element["memory"]
        assert mem.dtype == np.float32 and mem.ndim == 3 and mem.shape[1:] == (2, TOKEN_DIM), mem.shape
        for key in ("observation/image", "observation/wrist_image", "observation/state", "prompt"):
            assert key in element, key
        self.calls.append(element)
        return {"actions": np.zeros((50, 8), dtype=np.float32)}

    def reseed(self):
        self.reseeds += 1


def make_serve(**kw):
    pol = FakePolicy()
    return pol, serve.NativeMemServePolicy(pol, stub_encode, stub_online_frame, reseed_fn=pol.reseed, **kw)


def element(imgs, wrists, states):
    return {"observation/image": imgs[-1], "observation/wrist_image": wrists[-1],
            "observation/state": states[-1], "prompt": "pick up the cube"}


def check_memory_decodes(memory: np.ndarray, indices: list[int]):
    """两路分别还原：第 k 行 front token == front_value(indices[k])，wrist 同理。"""
    assert memory.shape == (len(indices), 2, TOKEN_DIM)
    for k, f in enumerate(indices):
        assert np.all(memory[k, 0] == front_value(f)), (k, f, memory[k, 0, 0])
        assert np.all(memory[k, 1] == wrist_value(f)), (k, f, memory[k, 1, 0])
    flat = memory.reshape(-1, TOKEN_DIM)             # 帧外层、视角内层
    assert np.all(flat[0::2, 0] == [front_value(f) for f in indices])
    assert np.all(flat[1::2, 0] == [wrist_value(f) for f in indices])


# ── 纯规则 ──────────────────────────────────────────────────────────────

def test_memory_frame_indices_rule():
    assert serve.memory_frame_indices(25, 12) == [0, 1, 13, 25]
    assert serve.memory_frame_indices(5, 12) == [0, 5]
    assert serve.memory_frame_indices(53, 12) == [0, 5, 17, 29, 41, 53]
    assert serve.memory_frame_indices(0, 12) == [0]
    assert serve.memory_frame_indices(48, 12) == [0, 12, 24, 36, 48]


# ── 状态机（不经 websocket） ──────────────────────────────────────────────

def _add(sv, start, n, exec_start_idx=0):
    imgs, wrists, states = frames(start, n)
    sv.add_buffer(client_utils.pack_buffer(imgs, states, exec_start_idx, wrist_image_buffer=wrists))
    return imgs, wrists, states


def test_two_rounds_with_demo_prefix_d5():
    pol, sv = make_serve()
    D = 5                                            # 演示 5 帧 + 当前帧 → 第一条 add_buffer 6 帧，t=D
    el = element(*_add(sv, 0, D + 1, exec_start_idx=D))
    out = sv.infer(el)
    assert list(out["memory_frames"]) == [0, 5]
    check_memory_decodes(pol.calls[-1]["memory"], [0, 5])
    el = element(*_add(sv, D + 1, 48))
    out = sv.infer(el)
    assert list(out["memory_frames"]) == [0, 5, 17, 29, 41, 53]
    assert out["memory_rows"] == 12
    check_memory_decodes(pol.calls[-1]["memory"], [0, 5, 17, 29, 41, 53])
    # 缓存增量复用：总共只编码过 6 帧 × 2 路
    assert sv.encode_calls == 12
    assert sorted(sv.tokens) == [0, 5, 17, 29, 41, 53]


def test_repeat_infer_does_not_advance():
    pol, sv = make_serve()
    el = element(*_add(sv, 0, 30))
    first = list(sv.infer(el)["memory_frames"])
    calls = sv.encode_calls
    second = list(sv.infer(el)["memory_frames"])
    assert first == second == [0, 5, 17, 29]
    assert sv.num_frames == 30 and sv.encode_calls == calls


def test_reset_clears_cache_and_frame_counter():
    pol, sv = make_serve()
    sv.infer(element(*_add(sv, 0, 20)))
    assert sv.tokens and sv.num_frames == 20
    sv.reset()
    assert sv.tokens == {} and sv.raw_frames == {} and sv.num_frames == 0 and sv.last_memory_frames == []
    assert pol.reseeds == 1
    # 新一局从第 0 帧重新计数：送入的像素按新局帧号编码，不会命中上一局缓存
    out = sv.infer(element(*_add(sv, 0, 3)))
    assert list(out["memory_frames"]) == [0, 2]
    check_memory_decodes(pol.calls[-1]["memory"], [0, 2])


def test_memory_over_640_rows_raises():
    pol, sv = make_serve()
    # t=3828 → 320 帧 × 2 = 640 行，允许；t=3840 → 321 帧 × 2 = 642 行，报错
    _add(sv, 0, 3829)
    out = sv.infer(element(*frames(3828, 1)))
    assert out["memory_rows"] == 640
    _add(sv, 3829, 12)
    with pytest.raises(ValueError, match="超过上限 640"):
        sv.infer(element(*frames(3840, 1)))


def test_missing_wrist_images_raises():
    pol, sv = make_serve()
    imgs, wrists, states = frames(0, 3)
    with pytest.raises(ValueError, match="wrist_images"):
        sv.add_buffer(client_utils.pack_buffer(imgs, states, 0))


def test_non_multiple_of_interval_exec_raises_instead_of_zero_fill():
    pol, sv = make_serve()
    sv.infer(element(*_add(sv, 0, 6)))              # t=5，缓存 {0,5}，丢弃 1..4 原图
    _add(sv, 6, 5)                                   # 执行 5 步（非 12 倍数）→ t=10 需要帧 10、0；10 在 → 正常
    sv.infer(element(*frames(10, 1)))
    _add(sv, 11, 7)                                  # t=17 需要 5、17 → 5 在缓存，正常
    assert list(sv.infer(element(*frames(17, 1)))["memory_frames"]) == [0, 5, 17]
    _add(sv, 18, 4)                                  # t=21 需要 9（已丢弃）→ 报错
    with pytest.raises(ValueError, match="没有原图"):
        sv.infer(element(*frames(21, 1)))


# ── 哨兵端到端：真实 pack_buffer → openpi_client websocket → 服务器 ──────────────

class _ServerThread:
    def __init__(self, sv):
        self.srv = serve.NativeMemWebsocketServer(sv, host="127.0.0.1", port=0, metadata={"fake": True})
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def main():
            self.stop = asyncio.Event()
            ready = asyncio.Event()
            task = asyncio.ensure_future(self.srv.run(ready=ready, stop=self.stop))
            await ready.wait()
            self._ready.set()
            await task

        self.loop.run_until_complete(main())

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(30), "服务器未就绪"
        return self.srv.bound_port

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.stop.set)
        self._thread.join(30)


def test_client_wrist_e2e_sentinel():
    import urllib.request

    ws_client = _load("nm_policy_ws_client", POLICY_WS_CLIENT)
    pol, sv = make_serve()
    with _ServerThread(sv) as port:
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10).read() == b"OK\n"

        # 第一局：照 eval.py 顺序 reset → add_buffer(含演示帧) → infer → 执行 48 步 → add_buffer → infer
        client = ws_client.MMEVLAWebsocketClientPolicy("127.0.0.1", port)
        assert client.get_server_metadata() == {"fake": True}
        assert client.reset()["reset_finished"] is True
        D = 5
        imgs, wrists, states = frames(0, D + 1)
        resp = client.add_buffer(client_utils.pack_buffer(imgs, states, D, wrist_image_buffer=wrists))
        assert resp["add_buffer_finished"] is True
        out = client.infer(element(imgs, wrists, states))
        assert out["actions"].shape == (50, 8)
        assert list(out["memory_frames"]) == [0, 5]
        check_memory_decodes(pol.calls[-1]["memory"], [0, 5])
        # 当前帧观测原样到达策略（前视、腕部不串）
        assert np.all(pol.calls[-1]["observation/image"] == front_value(D))
        assert np.all(pol.calls[-1]["observation/wrist_image"] == wrist_value(D))

        imgs, wrists, states = frames(D + 1, 48)
        client.add_buffer(client_utils.pack_buffer(imgs, states, 0, wrist_image_buffer=wrists))
        out = client.infer(element(imgs, wrists, states))
        indices = [0, 5, 17, 29, 41, 53]
        assert list(out["memory_frames"]) == indices
        check_memory_decodes(pol.calls[-1]["memory"], indices)
        # 重复 infer 不追加帧
        out = client.infer(element(imgs, wrists, states))
        assert list(out["memory_frames"]) == indices and sv.num_frames == D + 1 + 48
        client._ws.close()

        # 第二局：新 client、reset 后缓存为空、帧号归零
        client = ws_client.MMEVLAWebsocketClientPolicy("127.0.0.1", port)
        client.reset()
        assert sv.tokens == {} and sv.num_frames == 0
        imgs, wrists, states = frames(0, 26)
        client.add_buffer(client_utils.pack_buffer(imgs, states, 25, wrist_image_buffer=wrists))
        out = client.infer(element(imgs, wrists, states))
        assert list(out["memory_frames"]) == [0, 1, 13, 25]
        check_memory_decodes(pol.calls[-1]["memory"], [0, 1, 13, 25])
        client._ws.close()
    assert pol.reseeds == 2
    print("CLIENT_WRIST_E2E=PASS episodes=2 infers=5 frames_ep1=[0,5]→[0,5,17,29,41,53] frames_ep2=[0,1,13,25]")


# ── check_online_token.py 管线（打桩 S1 编码器；真实 V17 pkl 与 BinFill 前缀库只读）──────────────

def test_check_online_token_pipeline_with_stub_encoder(tmp_path, monkeypatch):
    import types

    import cv2
    import h5py

    from openpi.robomme import mem_encoder

    cot = _load("nm_check_online_token", HERE / "check_online_token.py")
    if not (cot.PKL_DIR.is_dir() and (cot.DEMO_STORE / "index.json").is_file()):
        pytest.skip("本机没有 V17 pkl 或 BinFill 前缀库")
    frames_doc = json.loads(cot.DEFAULT_FRAMES.read_text(encoding="utf-8"))
    assert len(frames_doc["exec_frames"]) == 32 and frames_doc["view_order"] == ["front", "wrist"]

    # 只打桩权重加载与编码（真实编码器的 CPU 小样另见 check_online_token.py run --g9a-only）；
    # offline_frame / online_frame 与 extract_mem_tokens.encode_episode_chunk 都用 S1 真实实现
    def encode_frames(enc, x):
        x = np.asarray(x, dtype=np.float32)
        return np.repeat(x.mean(axis=(1, 2, 3))[:, None], TOKEN_DIM, axis=1) / 255.0 + np.arange(TOKEN_DIM) * 1e-3

    monkeypatch.setattr(mem_encoder, "load_encoder", lambda path: object())
    monkeypatch.setattr(mem_encoder, "encode_frames", encode_frames)

    # 假每集 H5：只在冻结帧号放「前视 / 腕部」两张不同的 224 JPEG
    ep_dir = tmp_path / "data" / "episodes"
    ep_dir.mkdir(parents=True)
    by_ep = {}
    for fr in frames_doc["exec_frames"]:
        by_ep.setdefault(fr["global_episode_idx"], []).append(fr["t"])
    for g, ts in by_ep.items():
        n = max(ts) + 1
        bufs = {}
        for v, val in (("front", 60), ("wrist", 190)):
            ok, enc = cv2.imencode(".jpg", np.full((224, 224, 3), val, np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            arr = np.empty(n, dtype=object)
            for t in range(n):
                arr[t] = enc.reshape(-1)
            bufs[v] = arr
        with h5py.File(ep_dir / f"ep_{g:04d}.h5", "w") as h:
            h.attrs["num_timesteps"] = n
            for v in ("front", "wrist"):
                h.create_dataset(f"{v}_jpeg", data=bufs[v], dtype=h5py.vlen_dtype(np.uint8))
    report = tmp_path / "report.json"
    args = types.SimpleNamespace(frames=str(cot.DEFAULT_FRAMES), data_root=str(tmp_path / "data"), memtokens="",
                                 encoder_params="stub", max_exec_frames=0, g9a_only=False,
                                 report=str(report))
    assert cot.run(args) == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["g9a_pass"] and rep["g9a"]["n"] == 64 and rep["g9a"]["max_abs"] <= 1e-5
    assert set(rep["g9b_strata"]) == {"exec/front", "exec/wrist", "binfill_prefix/front", "binfill_prefix/wrist"}
    assert rep["n_exec_frames"] == 32 and rep["n_prefix_frames"] == 8
    # 反例：尺度差 1%（权重或预处理尺度错误）时 G9a 必须 FAIL；完全相同时 PASS
    s = cot.pair_stats(np.ones((4, 8)), np.ones((4, 8)) * 1.01)
    assert not cot.g9a_verdict(s) and cot.g9a_verdict(cot.pair_stats(np.ones((4, 8)), np.ones((4, 8))))


def test_check_online_token_manifest_pin_mismatch_refuses(tmp_path):
    cot = _load("nm_check_online_token_pin", HERE / "check_online_token.py")
    frames_doc = json.loads(cot.DEFAULT_FRAMES.read_text(encoding="utf-8"))
    if not pathlib.Path(frames_doc["sources"]["episode_manifest"]).is_file():
        pytest.skip("本机没有 V17 episode_manifest")
    assert cot.check_manifest_pin(frames_doc) == frames_doc["sources"]["episode_manifest_sha256"]
    bad = json.loads(json.dumps(frames_doc))
    bad["sources"]["episode_manifest_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="MANIFEST_PIN=FAIL"):
        cot.check_manifest_pin(bad)


# ── nm_trigger.sh：V17 日志末行 EXIT_CODE 必须写完整（审查 F1）──────────────────────────

def _trigger_env(tmp_path):
    import os
    fsmi = tmp_path / "fsmi"
    fsmi.write_text('#!/usr/bin/env bash\nfor i in 0 1 2 3 4 5 6 7; do echo "$i, 5"; done\n')
    fsmi.chmod(0o755)
    (tmp_path / "nm" / "runs").mkdir(parents=True)
    (tmp_path / "nm" / "runs" / "READY.json").write_text("{}")
    env = dict(os.environ, NM_ROOT=str(tmp_path / "nm"), V17_SESSION="s2-test-nonexistent-session",
               V17_LOG=str(tmp_path / "v17.log"), NVIDIA_SMI=str(fsmi), POLL_S="1",
               NM_TRIG_CMD="echo STEP4_GO")
    return env


def _run_trigger(env, *args, timeout=30):
    import subprocess
    return subprocess.run(["bash", str(HERE / "nm_trigger.sh"), *args], env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_nm_trigger_truncated_exit_code(tmp_path):
    import threading
    import time

    env = _trigger_env(tmp_path)
    log = tmp_path / "v17.log"
    head = "LOCAL692_START mode=eval run=local692-ctx-motion160-50k\n"
    # 截断：末行 EXIT_CODE=1 但没有换行，单轮（--once）不能判完成
    log.write_text(head + "EXIT_CODE=1")
    r = _run_trigger(env, "--once")
    assert r.returncode == 1 and "exit_code_unsettled(EXIT_CODE=1)" in r.stdout, r.stdout
    # 写完整（带换行）→ 单轮即成立
    log.write_text(head + "EXIT_CODE=0\n")
    r = _run_trigger(env, "--once")
    assert r.returncode == 0 and "newline=1" in r.stdout, r.stdout
    # 数字还在增长：第 1 轮读到 EXIT_CODE=1，0.5 s 后补成 EXIT_CODE=13（仍无换行）；
    # 第 2 轮末行变了 → 仍不成立；第 3 轮与第 2 轮相同 → 成立，起跑时末行必须是 EXIT_CODE=13
    log.write_text(head + "EXIT_CODE=1")
    threading.Timer(0.5, lambda: log.write_text(head + "EXIT_CODE=13")).start()
    r = _run_trigger(env)
    assert r.returncode == 0, r.stdout
    fired = [l for l in r.stdout.splitlines() if "NM_TRIGGER=FIRED" in l]
    assert len(fired) == 1 and "done(EXIT_CODE=13 newline=0 stable=1)" in fired[0], r.stdout
    assert "STEP4_GO" in r.stdout and "NM_TRIGGER_CMD_EXIT=0" in r.stdout
    # 锁已存在 → 再次运行拒绝
    r = _run_trigger(env)
    assert r.returncode == 3 and "NM_TRIGGER=REFUSED reason=lock_exists" in r.stdout
