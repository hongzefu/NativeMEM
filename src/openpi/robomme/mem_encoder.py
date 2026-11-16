"""NativeMEM Stage 1 记忆编码器（SiglipEncoder So400m/14）的加载与编码。

实现照搬官方 examples/robotwin/convert_raw_to_lerobot_mem_tokenizer.py 的 `_MemTokenExtractor`／
`MemTokenExtractor`：只构建 encoder 子树，从 Stage 1 params 里按 `_merge_params(missing_regex=".*")`
取 encoder/*，输入 /255*2-1 后取 query 输出 `x[:, :1]`，每张图得到一个 1152 维 token。

两条像素路径（S1 离线抽取与 S2 在线服务器共用，差异只记录不拦截，见计划 X 表与 G9b）：
- `offline_frame`：训练／离线抽取路径。每集 H5 里存的是 resize_with_pad_cv2 224 + JPEG q95 的字节，
  cv2.imdecode 后通道顺序原样（原版对 RGB 数组直接 imencode，解码回来仍是 RGB），不做颜色转换。
- `online_frame`：部署路径，照 NativeMEM-RMBench pi_model.py：openpi_client.image_tools.resize_with_pad
  到 224×224，不做 JPEG。
"""

from __future__ import annotations

import dataclasses
import logging

import cv2
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools

from openpi.models import model as _model
import openpi.models.siglip as _siglip
from openpi.models.siglip_encoder import SiglipEncoder
from openpi.shared import array_typing as at
import openpi.shared.download as download
from openpi.training import weight_loaders

SIZE = 224
SIGLIP_VARIANT = "So400m/14"
# 官方转换脚本 Args.dtype 默认 "float32"（MemTokenizerConfig.dtype 默认同为 float32）。
ENCODER_DTYPE = "float32"
TOKEN_DIM = _siglip.decode_variant(SIGLIP_VARIANT)["width"]  # 1152


class _MemTokenExtractor(nnx.Module):
    """与官方转换脚本同名类逐行一致：只含 encoder 子树。"""

    def __init__(self, siglip_variant: str, dtype: str, rngs: nnx.Rngs):
        siglip_params = _siglip.decode_variant(siglip_variant)
        encoder = nnx_bridge.ToNNX(
            SiglipEncoder(
                patch_size=siglip_params["patch_size"],
                width=siglip_params["width"],
                depth=siglip_params["depth"],
                mlp_dim=siglip_params["mlp_dim"],
                num_heads=siglip_params["num_heads"],
                scan=True,
                dtype_mm=dtype,
            )
        )
        encoder.lazy_init(jnp.zeros((1, SIZE, SIZE, 3), dtype=jnp.float32), train=False, rngs=rngs)
        self.encoder = encoder


@dataclasses.dataclass
class MemEncoder:
    """load_encoder 的返回值：已加载 Stage 1 encoder 权重的模块及其来源。"""

    model: _MemTokenExtractor
    params_path: str
    token_dim: int = TOKEN_DIM


def _stringify_keys(d):
    if isinstance(d, dict):
        return {str(k): _stringify_keys(v) for k, v in d.items()}
    return d


def _restore_keys(ref, merged):
    if not isinstance(ref, dict) or not isinstance(merged, dict):
        return merged
    result = {}
    ref_key_map = {str(k): k for k in ref}
    for k, v in merged.items():
        orig_key = ref_key_map.get(k, k)
        result[orig_key] = _restore_keys(ref.get(orig_key, {}), v)
    return result


def load_encoder(params_path: str) -> MemEncoder:
    """从 Stage 1 params（…/pretrain/49999/params）的 encoder 子树构建 SiglipEncoder So400m/14。

    缺少 encoder 子树、或 encoder 有任何参数没被权重覆盖时抛错（不允许静默随机初始化）。
    """
    model = _MemTokenExtractor(SIGLIP_VARIANT, ENCODER_DTYPE, rngs=nnx.Rngs(0))
    path = download.maybe_download(params_path)
    loaded = _model.restore_params(path, restore_type=np.ndarray)
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    if not any(k.startswith("encoder/") for k in flat_loaded):
        raise ValueError(f"{params_path} 缺少 encoder 子树，不是 mem_tokenizer Stage 1 存档")

    graphdef, state = nnx.split(model)
    params_shape = state.to_pure_dict()
    flat_ref = flax.traverse_util.flatten_dict(_stringify_keys(params_shape), sep="/")
    # 只保留 encoder 子树，其余（PaliGemma、动作头等）与本模块无关。
    enc_loaded = flax.traverse_util.unflatten_dict(
        {k: v for k, v in flat_loaded.items() if k.startswith("encoder/")}, sep="/"
    )
    missing = [
        k for k in flat_ref if k not in flat_loaded and (k + "/value") not in flat_loaded
    ]
    if missing:
        raise ValueError(f"encoder 有 {len(missing)} 个参数在存档里找不到，例如 {missing[:3]}")
    merged = weight_loaders._merge_params(enc_loaded, _stringify_keys(params_shape), missing_regex=".*")
    merged = _restore_keys(params_shape, merged)
    at.check_pytree_equality(expected=params_shape, got=merged, check_shapes=True, check_dtypes=False)
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, state)
    model.eval()
    logging.info("已加载记忆编码器 %s（%d 个参数张量）", params_path, len(flat_ref))
    return MemEncoder(model=model, params_path=str(params_path))


@nnx.jit
def _encode(model: _MemTokenExtractor, images: jax.Array) -> jax.Array:
    mem_token, _, _ = model.encoder(images, train=False, extract_block=None, ids_keep=None)
    return jax.lax.stop_gradient(mem_token[:, 0])


def encode_frames(enc: MemEncoder, frames: np.ndarray) -> np.ndarray:
    """uint8[B,224,224,3] → f32[B,1152]。

    归一化与官方 MemTokenExtractor._prepare_images 相同：image_tools.resize_with_pad（224→224 恒等）
    后 /255*2-1；取 query 输出 x[:, :1]。不做 batch 补齐，调用方自行决定 batch 形状（重编译代价由调用方控制）。
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (SIZE, SIZE, 3):
        raise ValueError(f"encode_frames 需要 uint8[B,{SIZE},{SIZE},3]，得到 {frames.dtype} {frames.shape}")
    images = image_tools.resize_with_pad(frames, SIZE, SIZE).astype(np.float32) / 255.0 * 2.0 - 1.0
    return np.asarray(_encode(enc.model, jnp.asarray(images)), dtype=np.float32)


def offline_frame(jpeg_bytes) -> np.ndarray:
    """训练与离线抽取路径：每集 H5 的 JPEG 字节 → uint8[224,224,3]（cv2.imdecode，通道顺序原样）。"""
    buf = np.frombuffer(jpeg_bytes, np.uint8) if isinstance(jpeg_bytes, bytes) else np.asarray(jpeg_bytes, np.uint8).reshape(-1)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.shape != (SIZE, SIZE, 3):
        raise ValueError(f"JPEG 解码失败或尺寸不是 {SIZE}×{SIZE}：{None if img is None else img.shape}")
    return img


def online_frame(raw_uint8_hw3: np.ndarray) -> np.ndarray:
    """部署路径（照原版 pi_model.py）：resize_with_pad 到 224×224，不做 JPEG。"""
    raw = np.asarray(raw_uint8_hw3)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"online_frame 需要 uint8[H,W,3]，得到 {raw.dtype} {raw.shape}")
    return np.asarray(image_tools.resize_with_pad(raw, SIZE, SIZE), dtype=np.uint8)
