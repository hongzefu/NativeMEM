"""训练数值有限性闸门（计划 J#9：SMOKE20／TRAIN_FINAL 的 finite=PASS）与续训核对（J#19 G8b RESUME）。

子检查（可组合，全部通过才 finite=PASS）：
  --log <train.log>        日志里每一行 "Step N: loss=…, grad_norm=…, param_norm=…" 都必须有限；至少 1 行
  --ckpt <ckpt>/<step>     存档 params（EMA）与 train_state（raw params、Adam 状态）全部叶子有限
  --infer <ckpt>/<step>    用该存档建策略，对固定 4 个真实观测推理，输出 actions 形状 (50,8) 且全有限
  --selftest               向合成日志、合成参数树、合成动作分别注入 NaN／inf，必须全部判 FAIL；干净样本判 PASS
另有续训核对：
  --resume-check <ckpt_dir> --from-step 19 --to-step 21
      to 存档的 step == to+1、Adam count 全等于 step（证明 Adam 状态从 from 存档恢复，而不是从头初始化）、
      EMA 已恢复（‖ema_to − ema_from‖∞ ≤ ‖raw_to − ema_from‖∞ 且 ema_to ≠ raw_to）。输出 RESUME=PASS|FAIL。
末行：CHECK_FINITE finite=PASS|FAIL <各子项>=PASS|FAIL
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

STEP_RE = re.compile(r"Step (\d+): (.*)$")
# train_nativemem.py 的日志格式 "%(message)-80s (%(process)d:%(filename)s:%(lineno)s)"，尾部要剥掉。
TAIL_RE = re.compile(r"\s+\(\d+:[^()]*:\d+\)\s*$")
REQUIRED_KEYS = ("loss", "grad_norm", "param_norm")


def check_log_text(text: str) -> tuple[bool, str]:
    n = 0
    bad = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        n += 1
        kv = {}
        for part in TAIL_RE.sub("", m.group(2)).split(","):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                kv[k.strip()] = v.strip()
        for k in REQUIRED_KEYS:
            v = kv.get(k)
            try:
                ok = v is not None and math.isfinite(float(v))
            except ValueError:
                ok = False
            if not ok:
                bad.append(f"step{m.group(1)}:{k}={v}")
    if n == 0:
        return False, "steps=0"
    return (not bad), f"steps={n} bad={len(bad)}" + (f" first={bad[0]}" if bad else "")


def check_tree_finite(tree) -> tuple[bool, str]:
    import jax

    n = nonfinite = 0
    first = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        arr = np.asarray(leaf)
        if not (np.issubdtype(arr.dtype, np.floating) or arr.dtype.name == "bfloat16"):
            continue
        n += 1
        if not np.all(np.isfinite(arr.astype(np.float32))):
            nonfinite += 1
            first = first or jax.tree_util.keystr(path)
    if n == 0:
        return False, "leaves=0"
    return nonfinite == 0, f"leaves={n} nonfinite={nonfinite}" + (f" first={first}" if first else "")


def check_actions(actions) -> tuple[bool, str]:
    a = np.asarray(actions)
    ok = a.shape == (50, 8) and bool(np.all(np.isfinite(a)))
    return ok, f"shape={a.shape} finite={bool(np.all(np.isfinite(a)))}"


def restore_tree_numpy(path: str | pathlib.Path):
    """把一个 orbax PyTree 目录整体读成 numpy 树（与 openpi.models.model.restore_params 同法，但不限 params 键）。"""
    import jax
    import orbax.checkpoint as ocp

    path = pathlib.Path(path).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        meta = ckptr.metadata(path)
        meta = meta.tree if hasattr(meta, "tree") else meta
        item = {k: meta[k] for k in meta}
        return ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item),
            ),
        )


def check_ckpt(step_dir: str | pathlib.Path) -> tuple[bool, str]:
    step_dir = pathlib.Path(step_dir)
    msgs, ok = [], True
    for item in ("params", "train_state"):
        p = step_dir / item
        if not p.exists():
            return False, f"missing={item}"
        good, msg = check_tree_finite(restore_tree_numpy(p))
        ok &= good
        msgs.append(f"{item}:{msg}")
    return ok, " ".join(msgs)


def check_infer(step_dir: str | pathlib.Path, config_name: str, n: int = 4) -> tuple[bool, str]:
    import openpi.shared.normalize as _normalize
    from openpi.policies import policy_config
    from openpi.robomme.dataset import RoboMMEMemDataset
    from openpi.training import config_nativemem

    step_dir = pathlib.Path(step_dir)
    cfg = config_nativemem.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _normalize.load(step_dir / "assets" / data_cfg.asset_id)
    policy = policy_config.create_trained_policy(cfg, step_dir, norm_stats=norm_stats)
    ds = RoboMMEMemDataset(
        data_cfg.episodes_dir, data_cfg.manifest_path, data_cfg.memmap_path, action_horizon=cfg.model.action_horizon
    )
    idxs = [int(round(i * (len(ds) - 1) / max(n - 1, 1))) for i in range(n)]
    msgs, ok = [], True
    for i in idxs:
        item = ds[i]
        obs = {k: item[k] for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt", "memory")}
        good, msg = check_actions(policy.infer(obs)["actions"])
        ok &= good
        msgs.append(f"idx{i}:{msg}")
    return ok, " ".join(msgs)


def _flat_numeric(tree) -> dict[str, np.ndarray]:
    import jax

    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_leaves_with_path(tree)}


def resume_check(ckpt_dir: str | pathlib.Path, from_step: int, to_step: int) -> tuple[bool, str]:
    ckpt_dir = pathlib.Path(ckpt_dir)
    ts_to = _flat_numeric(restore_tree_numpy(ckpt_dir / str(to_step) / "train_state"))
    step_vals = [v for k, v in ts_to.items() if k.endswith("['step']")]
    counts = [v for k, v in ts_to.items() if "count" in k and np.issubdtype(v.dtype, np.integer)]
    if len(step_vals) != 1:
        return False, f"step_leaf={len(step_vals)}"
    step = int(step_vals[0])
    reasons = []
    if step != to_step + 1:
        reasons.append(f"step={step}!={to_step + 1}")
    if not counts or any(int(c) != step for c in counts):
        reasons.append(f"adam_count={[int(c) for c in counts][:4]}")
    ema_to = _flat_numeric(restore_tree_numpy(ckpt_dir / str(to_step) / "params"))
    ema_from = _flat_numeric(restore_tree_numpy(ckpt_dir / str(from_step) / "params"))
    raw_to = {k.replace("['params']", "", 1): v for k, v in ts_to.items() if k.startswith("['params']")}
    d_ema = d_raw = d_er = 0.0
    matched = 0
    for k, e_to in ema_to.items():
        kk = k.replace("['params']", "", 1)
        if k not in ema_from or kk not in raw_to or not np.issubdtype(e_to.dtype, np.floating):
            continue
        matched += 1
        ef = ema_from[k].astype(np.float32)
        et = e_to.astype(np.float32)
        rt = raw_to[kk].astype(np.float32)
        d_ema = max(d_ema, float(np.max(np.abs(et - ef))))
        d_raw = max(d_raw, float(np.max(np.abs(rt - ef))))
        d_er = max(d_er, float(np.max(np.abs(et - rt))))
    if matched == 0:
        reasons.append("ema_unmatched")
    elif not (d_ema <= d_raw and d_er > 0):
        reasons.append(f"ema d_ema={d_ema:.3e} d_raw={d_raw:.3e} d_ema_raw={d_er:.3e}")
    ok = not reasons
    return ok, (
        f"RESUME={'PASS' if ok else 'FAIL'} from={from_step} to={to_step} step={step} adam_counts={len(counts)} "
        f"ema_leaves={matched} d_ema={d_ema:.3e} d_raw={d_raw:.3e}" + (f" reasons={';'.join(reasons)}" if reasons else "")
    )


def selftest() -> tuple[bool, str]:
    import orbax.checkpoint as ocp

    results = []
    clean_log = (
        "Step 0: grad_norm=1.2, loss=0.5, param_norm=300.0\n"
        "12:00:00.000 [I] Step 1: loss=0.4000, grad_norm=1.1000, param_norm=300.1000"
        + " " * 20 + " (12345:train_nativemem.py:271)\n"
    )
    results.append(("log_clean", check_log_text(clean_log)[0], True))
    results.append(("log_nan", check_log_text(clean_log + "Step 2: grad_norm=nan, loss=0.4, param_norm=300.1\n")[0], False))
    results.append(("log_inf", check_log_text(clean_log + "Step 2: grad_norm=1.0, loss=inf, param_norm=300.1\n")[0], False))
    results.append(("log_empty", check_log_text("no steps here\n")[0], False))
    clean = {"a": {"w": np.ones((3, 4), np.float32)}, "b": np.zeros(2, np.float32), "step": np.int32(3)}
    dirty = {"a": {"w": np.ones((3, 4), np.float32)}, "b": np.array([0.0, np.nan], np.float32), "step": np.int32(3)}
    results.append(("tree_clean", check_tree_finite(clean)[0], True))
    results.append(("tree_nan", check_tree_finite(dirty)[0], False))
    with tempfile.TemporaryDirectory() as td:
        for name, tree in (("clean", clean), ("dirty", dirty)):
            for item in ("params", "train_state"):
                with ocp.PyTreeCheckpointer() as ck:
                    ck.save(pathlib.Path(td) / name / item, tree)
        results.append(("ckpt_clean", check_ckpt(pathlib.Path(td) / "clean")[0], True))
        results.append(("ckpt_nan", check_ckpt(pathlib.Path(td) / "dirty")[0], False))
    a = np.zeros((50, 8), np.float32)
    results.append(("actions_clean", check_actions(a)[0], True))
    a_bad = a.copy()
    a_bad[3, 2] = np.nan
    results.append(("actions_nan", check_actions(a_bad)[0], False))
    results.append(("actions_shape", check_actions(np.zeros((49, 8)))[0], False))
    wrong = [name for name, got, want in results if got != want]
    return not wrong, f"cases={len(results)} wrong={wrong}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--ckpt")
    ap.add_argument("--infer")
    ap.add_argument("--config", default="nativemem_pi05_robomme4task")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resume-check")
    ap.add_argument("--from-step", type=int, default=19)
    ap.add_argument("--to-step", type=int, default=21)
    args = ap.parse_args()

    if args.resume_check:
        ok, line = resume_check(args.resume_check, args.from_step, args.to_step)
        print(line, flush=True)
        return 0 if ok else 1

    parts, all_ok = [], True
    if args.selftest:
        ok, msg = selftest()
        print(f"[selftest] {msg}", flush=True)
        parts.append(f"selftest={'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    if args.log:
        ok, msg = check_log_text(pathlib.Path(args.log).read_text(errors="replace"))
        print(f"[log] {msg}", flush=True)
        parts.append(f"log={'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    if args.ckpt:
        ok, msg = check_ckpt(args.ckpt)
        print(f"[ckpt] {msg}", flush=True)
        parts.append(f"ckpt={'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    if args.infer:
        ok, msg = check_infer(args.infer, args.config)
        print(f"[infer] {msg}", flush=True)
        parts.append(f"infer={'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    if not parts:
        ap.error("至少给一个检查项")
    print(f"CHECK_FINITE finite={'PASS' if all_ok else 'FAIL'} " + " ".join(parts), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
