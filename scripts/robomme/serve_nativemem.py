"""NativeMEM 4task 评估服务器（1116 计划 C2 第 2 行；J#13 #14 #16 #17）。

协议与 policy 仓 ``src/mme_vla_suite/serving/websocket_policy_server.py`` 一致：
连上先发 metadata；随后处理三类消息——``{"reset": True}``、``{"add_buffer": True, ...}``、其余一律当 infer；
另有 HTTP ``/healthz``。客户端是 ``scripts/robomme/client/eval.py``（policy 仓客户端的最小改动副本）。

内部逻辑照 NativeMEM-RMBench ``policy/nativemem/pi_model.py``：
- 帧号：只有 add_buffer 推进帧号；reset 后第一条 add_buffer 的第一帧是第 0 帧（含演示帧）。infer 不追加帧。
- infer 时 t = 已收到的最后一帧号；记忆帧号 ``range(t, -1, -12)``，末尾不是 0 就补 0，再反转为升序
  （与训练侧 ``MemLeRobotDataset._get_memory_indices`` 同一规则，计划 X9：锚在当前帧）。
- 每帧两路（[front, wrist]），各 1 个 1152 维 token，按帧号缓存；缺 token 的帧按需编码，
  **逐帧逐路 batch=1**（原版 ``[1,224,224,3]`` 固定形状，避免每局按变长 batch 重编译）。启动时预热一次。
- 在线预处理照原版 ``openpi_client.image_tools.resize_with_pad`` 224、不做 JPEG（由 S1 的 ``online_frame`` 实现）。
- 记忆送策略的形状是 ``[T, 2, 1152]``（帧外层、视角内层；展平即 ``(2T, 1152)``）；展平行数 > 640 直接报错，不截断。
- reset 清空 token 缓存与原图、帧号归零，并把策略 rng 重新置为 ``jax.random.key(seed)``——
  这是 policy 仓 ``scripts/training/serve_policy.py --seed`` → ``MME_VLA_Policy.reset`` 的语义（J#14 要求按其执行）。

原图只为「将来可能用到」的帧保留：每次 infer 之后，帧号 <= t 且没有进入缓存的原图丢弃。
依据：客户端每次执行 48 步（12 的倍数），之后的 t' 满足 t' ≡ t (mod 12)，它所需的 <= t 的帧恰好已在缓存里；
若客户端执行步数不是 12 的倍数，所需帧会缺原图，infer 直接报错（不静默补零）。
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import http
import logging
import pathlib
import time
import traceback
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("serve_nativemem")

FORK_ROOT = pathlib.Path(__file__).resolve().parents[2]
NM_ROOT = pathlib.Path("/scratch/hongze/NativeMEM")
DEFAULT_ENCODER_PARAMS = NM_ROOT / "checkpoints/mem_tokenizer_pretrain/pretrain/49999/params"
ASSET_ID = "robomme4task"
VIEWS = ("front", "wrist")
TOKEN_DIM = 1152


def memory_frame_indices(t: int, interval: int) -> list[int]:
    """记忆帧号：range(t,-1,-interval)，末尾补 0，反转为升序。例：t=25,interval=12 → [0,1,13,25]。"""
    if t < 0:
        raise ValueError(f"当前帧号必须 >= 0，得到 {t}")
    indices = list(range(t, -1, -interval))
    if indices[-1] != 0:
        indices.append(0)
    indices.reverse()
    return indices


class NativeMemServePolicy:
    """服务器内部状态机：帧号、原图、token 缓存与对策略的调用。

    ``encode_fn(uint8[1,224,224,3]) -> f32[1,1152]``；``online_frame_fn(uint8[H,W,3]) -> uint8[224,224,3]``。
    ``policy`` 只需有 ``infer(dict) -> dict`` 与可选的 ``metadata``；``reseed_fn`` 在 reset 时调用。
    """

    def __init__(
        self,
        policy: Any,
        encode_fn: Callable[[np.ndarray], np.ndarray],
        online_frame_fn: Callable[[np.ndarray], np.ndarray],
        *,
        memory_interval: int = 12,
        max_mem_rows: int = 640,
        reseed_fn: Callable[[], None] | None = None,
    ) -> None:
        if memory_interval <= 0:
            raise ValueError(f"memory_interval 必须为正，得到 {memory_interval}")
        self._policy = policy
        self._encode_fn = encode_fn
        self._online_frame_fn = online_frame_fn
        self.memory_interval = memory_interval
        self.max_mem_rows = max_mem_rows
        self._reseed_fn = reseed_fn
        self.raw_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.tokens: dict[int, np.ndarray] = {}   # 帧号 -> f32[2,1152]（[front, wrist]）
        self.num_frames = 0
        self.encode_calls = 0                      # 累计 batch=1 编码调用次数（每路一次）
        self.last_memory_frames: list[int] = []

    @property
    def metadata(self) -> dict:
        return dict(getattr(self._policy, "metadata", {}) or {})

    def reset(self) -> None:
        self.raw_frames.clear()
        self.tokens.clear()
        self.num_frames = 0
        self.last_memory_frames = []
        if self._reseed_fn is not None:
            self._reseed_fn()

    def add_buffer(self, obs: dict) -> int:
        images = np.asarray(obs["images"])
        if "wrist_images" not in obs:
            raise ValueError("add_buffer 缺 wrist_images：客户端须用 NativeMEM 副本（pack_buffer 传 wrist_image_buffer）")
        wrists = np.asarray(obs["wrist_images"])
        if images.ndim == 5:          # 客户端 pack_buffer 形状 [N,1,H,W,3]
            images = images[:, 0]
        if wrists.ndim == 5:
            wrists = wrists[:, 0]
        if images.ndim != 4 or images.shape[-1] != 3 or wrists.shape != images.shape:
            raise ValueError(f"add_buffer 形状不符：images {images.shape} wrist_images {wrists.shape}")
        if images.dtype != np.uint8 or wrists.dtype != np.uint8:
            raise ValueError(f"add_buffer 需要 uint8，得到 {images.dtype}/{wrists.dtype}")
        for front, wrist in zip(images, wrists):
            self.raw_frames[self.num_frames] = (front, wrist)
            self.num_frames += 1
        return len(images)

    def _encode_one(self, frame_idx: int) -> np.ndarray:
        if frame_idx not in self.raw_frames:
            raise ValueError(
                f"记忆帧 {frame_idx} 没有原图（未收到或已丢弃；客户端执行步数须为 {self.memory_interval} 的倍数）")
        per_view = []
        for raw in self.raw_frames[frame_idx]:
            x = np.asarray(self._online_frame_fn(raw), dtype=np.uint8)[None]   # [1,224,224,3]
            tok = np.asarray(self._encode_fn(x), dtype=np.float32)
            self.encode_calls += 1
            if tok.shape != (1, TOKEN_DIM):
                raise ValueError(f"编码器输出形状应为 (1,{TOKEN_DIM})，得到 {tok.shape}")
            per_view.append(tok[0])
        return np.stack(per_view, axis=0)                                      # [2,1152]

    def build_memory(self) -> tuple[list[int], np.ndarray]:
        if self.num_frames == 0:
            raise ValueError("infer 之前没有任何 add_buffer 帧")
        t = self.num_frames - 1
        indices = memory_frame_indices(t, self.memory_interval)
        rows = len(indices) * len(VIEWS)
        if rows > self.max_mem_rows:
            raise ValueError(f"记忆行数 {rows} 超过上限 {self.max_mem_rows}（t={t}），按计划报错不截断")
        for i in indices:
            if i not in self.tokens:
                self.tokens[i] = self._encode_one(i)
        memory = np.stack([self.tokens[i] for i in indices], axis=0).astype(np.float32)   # [T,2,1152]
        # 丢弃 <= t 且未进缓存的原图（见模块说明）
        for f in [f for f in self.raw_frames if f <= t and f not in self.tokens]:
            del self.raw_frames[f]
        self.last_memory_frames = indices
        return indices, memory

    def infer(self, obs: dict) -> dict:
        indices, memory = self.build_memory()
        element = dict(obs)
        element["memory"] = memory
        out = dict(self._policy.infer(element))
        out["memory_frames"] = np.asarray(indices, dtype=np.int64)
        out["memory_rows"] = int(memory.shape[0] * memory.shape[1])
        return out


class NativeMemWebsocketServer:
    """websocket 外壳：消息分派照抄 policy 仓 WebsocketPolicyServer._handler。"""

    def __init__(self, policy: NativeMemServePolicy, host: str = "0.0.0.0", port: int | None = None,
                 metadata: dict | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self, ready: asyncio.Event | None = None, stop: asyncio.Event | None = None):
        import websockets.asyncio.server as _server

        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                 process_request=_health_check) as server:
            self.bound_port = next(iter(server.sockets)).getsockname()[1]
            if ready is not None:
                ready.set()
            if stop is None:
                await server.serve_forever()
            else:
                await stop.wait()

    async def _handler(self, websocket):
        import websockets
        import websockets.frames
        from openpi_client import msgpack_numpy

        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset()
                    await websocket.send(packer.pack(
                        {"reset_finished": True, "reset_time_ms": (time.monotonic() - tstart) * 1000}))
                elif obs.get("add_buffer", False):
                    tstart = time.monotonic()
                    n = self._policy.add_buffer(obs)
                    tend = time.monotonic() - tstart
                    logger.info("TIMING add_buffer_ms=%.1f frames=%d total_frames=%d exec_start_idx=%s",
                                tend * 1000, n, self._policy.num_frames, obs.get("exec_start_idx"))
                    await websocket.send(packer.pack(
                        {"add_buffer_finished": True, "add_buffer_time_ms": tend * 1000}))
                else:
                    tstart = time.monotonic()
                    outputs = self._policy.infer(obs)
                    logger.info("TIMING infer_ms=%.1f t=%d mem_frames=%d encode_calls=%d",
                                (time.monotonic() - tstart) * 1000, self._policy.num_frames - 1,
                                len(self._policy.last_memory_frames), self._policy.encode_calls)
                    await websocket.send(packer.pack(outputs))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="Internal server error. Traceback included in previous frame.")
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_real_policy(args) -> NativeMemServePolicy:
    """真实路径：加载 S1 配置、存档 norm_stats、作者编码器。只在 main 里调用（测试不走这里）。"""
    import jax

    from openpi.policies import policy_config
    from openpi.robomme import mem_encoder
    from openpi.shared import normalize
    from openpi.training import config_nativemem

    ckpt = pathlib.Path(args.ckpt).resolve()
    if not (ckpt / "params").is_dir():
        raise FileNotFoundError(f"存档缺 params/：{ckpt}")
    ns_dir = ckpt / "assets" / ASSET_ID
    ns_file = ns_dir / "norm_stats.json"
    if not ns_file.is_file():
        raise FileNotFoundError(f"存档缺 norm_stats：{ns_file}")
    ref = pathlib.Path(args.norm_ref)
    if not ref.is_file():
        raise FileNotFoundError(f"G4 归一化统计参照文件不存在：{ref}")
    sha, ref_sha = sha256_file(ns_file), sha256_file(ref)
    if sha != ref_sha:
        raise ValueError(f"存档 norm_stats sha256 {sha} != G4 产物 {ref_sha}（{ref}）")
    logger.info("NORM_SHA=%s file=%s ref=%s", sha[:8], ns_file, ref)
    print(f"NORM_SHA={sha[:8]} file={ns_file}", flush=True)
    norm_stats = normalize.load(ns_dir)

    cfg = config_nativemem.get_config(args.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, assets=dataclasses.replace(cfg.data.assets, assets_dir=str(ckpt / "assets"))))
    policy = policy_config.create_trained_policy(cfg, ckpt, norm_stats=norm_stats)

    def reseed() -> None:
        policy._rng = jax.random.key(args.seed)   # 与 policy 仓 MME_VLA_Policy.reset 同语义：每局重置为 key(seed)

    reseed()
    print(f"SERVE_RNG source=jax.random.key({args.seed}) reset=reseed (policy 仓 serve_policy.py --seed 语义)", flush=True)

    enc = mem_encoder.load_encoder(str(args.encoder_params))

    def encode_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(mem_encoder.encode_frames(enc, x))

    t0 = time.monotonic()
    warm = encode_fn(np.zeros((1, 224, 224, 3), dtype=np.uint8))   # 预热：固定形状 [1,224,224,3] 编译一次
    print(f"ENCODER_WARMUP shape={tuple(warm.shape)} seconds={time.monotonic() - t0:.1f} params={args.encoder_params}",
          flush=True)
    return NativeMemServePolicy(policy, encode_fn, mem_encoder.online_frame,
                                memory_interval=args.memory_interval, max_mem_rows=args.max_mem_rows,
                                reseed_fn=reseed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="存档目录（含 params/ 与 assets/robomme4task/norm_stats.json）")
    parser.add_argument("--config", default="nativemem_pi05_robomme4task")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--encoder-params", default=str(DEFAULT_ENCODER_PARAMS))
    parser.add_argument("--norm-ref", default=str(FORK_ROOT / "assets" / ASSET_ID / "norm_stats.json"),
                        help="G4 产物；存档里的 norm_stats.json 必须与它 sha256 相同")
    parser.add_argument("--memory-interval", type=int, default=12)
    parser.add_argument("--max-mem-rows", type=int, default=640)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    serve = build_real_policy(args)
    server = NativeMemWebsocketServer(serve, host=args.host, port=args.port, metadata=serve.metadata)
    print(f"SERVE_NATIVEMEM port={args.port} config={args.config} ckpt={args.ckpt} interval={args.memory_interval} "
          f"max_mem_rows={args.max_mem_rows}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
