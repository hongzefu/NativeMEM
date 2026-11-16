"""RoboMME（Panda 单臂 8 维）的 NativeMEM 输入输出变换。

照原版 AlohaInputs／AlohaOutputs 的结构改写（计划 X4）：
- 两路相机：front → base_0_rgb、wrist → left_wrist_0_rgb；right_wrist_0_rgb 用 np.zeros_like(base) 填、
  image_mask=False（与 AlohaInputs 缺失相机的处理相同）。
- state 8 维 = 7 个关节 ⊕ 夹爪，无关节翻转、无夹爪空间换算。
- RoboMMEOutputs 只取模型输出的前 8 维。
输出链顺序照原版：Unnormalize → AbsoluteActions(make_bool_mask(7,-1)) → RoboMMEOutputs。

训练与推理共用同一组输入键（不需要 repack）：
  observation/image、observation/wrist_image：uint8 [H,W,3]（训练 224，推理 256，之后统一 ResizeImages 224）
  observation/state：f32 [8]
  actions：f32 [50, 8]（只在训练时有）
  prompt：str
  memory：f32 [T, 2, 1152]（视角顺序 [front, wrist]）
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms

STATE_DIM = 8
ACTION_DIM = 8
# 前 7 维关节做 delta，夹爪（第 8 维）保持绝对；DeltaActions／AbsoluteActions 与归一化统计共用。
DELTA_ACTION_MASK = transforms.make_bool_mask(7, -1)


def _to_uint8_hwc(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"图像应为 [H,W,3]，得到 {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def make_robomme_example() -> dict:
    """推理输入示例（与 S2 服务器的输入键一致）。"""
    return {
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(STATE_DIM).astype(np.float32),
        "prompt": "pick up the cube",
        "memory": np.random.randn(3, 2, 1152).astype(np.float32),
    }


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    """RoboMME 两路相机、8 维单臂输入。"""

    def __call__(self, data: dict) -> dict:
        base_image = _to_uint8_hwc(data["observation/image"])
        wrist_image = _to_uint8_hwc(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32).copy()
        if state.shape != (STATE_DIM,):
            raise ValueError(f"observation/state 应为 ({STATE_DIM},)，得到 {state.shape}")

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": state,
        }

        # 动作只在训练时存在；copy 防止后续 DeltaActions 原地修改上游数据。
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"actions 应为 [H,{ACTION_DIM}]，得到 {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEMemInputs(transforms.DataTransformFn):
    """RoboMMEInputs 外加透传 memory（对应原版 AlohaMemInputs）。"""

    def __call__(self, data: dict) -> dict:
        memory = data.get("memory")
        out = RoboMMEInputs()(data)
        if memory is not None:
            memory = np.asarray(memory, dtype=np.float32)
            if memory.ndim != 3 or memory.shape[1] != 2:
                raise ValueError(f"memory 应为 [T,2,D]（视角 [front,wrist]），得到 {memory.shape}")
            out["memory"] = memory
        return out


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    """只返回前 8 维动作（7 关节绝对值 + 夹爪）。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
