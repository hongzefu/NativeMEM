"""从 4task-v2-1600ep 原始单 episode H5 重抽 NativeMEM Stage 2 训练数据（第二版，取代 build_memframes.py）。

每集输出 <out>/episodes/ep_<global_episode_idx:04d>.h5：
  front_jpeg  vlen uint8 [T]  每帧 obs/front_rgb（256×256 RGB）→ resize_with_pad_cv2 224 → cv2.imencode JPEG q95，RGB 原样
  wrist_jpeg  vlen uint8 [T]  每帧 obs/wrist_rgb，预处理同上
  state       f32 [T,8]       concat(obs/joint_state[t], obs/gripper_state[t][:1])
  action      f32 [T,8]       action/joint_action[t]（不移位、不丢帧）
  is_demo     bool [T]        info/is_video_demo
  attrs: global_episode_idx, task, difficulty, episode, seed, member, h5_sha256, num_timesteps,
         exec_start_idx, task_goal（setup/task_goal[0] 解码后小写）, complete=True

校验（全部通过才写 complete 并落盘；任一失败即抛错，不留输出）：
  0. 映射：job 的 global／输出文件名／pkl 路径／帧数／演示长度，与由 member 独立查 episode_map → manifest 得到的值逐条相等；
  1. 原始 H5 的 sha256 与字节数 == episode_map；
  2. 帧数 == manifest.num_timesteps，demo 前缀长度 == exec_start_idx；
  3. 第一个执行帧的原始 front_rgb／wrist_rgb／state 与 V17 pkl 的 image／wrist_image／state 逐字节相等，
     （写入的 state[exec] 与 pkl state 逐位相等放在第 4 项读回之后）；
  4. 派生值：写入临时文件后读回，每帧 state／action 与原始字段独立重算值逐位相等；首帧、尾帧、演示末帧、
     执行首帧 4 帧上前视与腕部分别用参考预处理重算 JPEG，解码像素与写入值逐位相等；is_demo 与 attrs 逐项相等。

写入逻辑拆成 select_views／derive_state／derive_action_seq／build_jobs 四个函数，单测用 monkeypatch 改写它们做故障注入；
校验部分不调用这四个函数，而是直接读原始字段独立重算。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np

SIZE = 224
QUALITY = 95
JPEG_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]


def resize_with_pad_cv2(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """与 NativeMEM scripts/prepare_mem_tokenizer_h5.py::resize_with_pad_cv2 逐行一致（同 build_memframes.py）。"""
    cur_h, cur_w = image.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return image
    scale = min(width / cur_w, height / cur_h)
    resized_w = max(1, round(cur_w * scale))
    resized_h = max(1, round(cur_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    pad_h = (height - resized_h) // 2
    pad_w = (width - resized_w) // 2
    out[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return out


def encode_jpeg(img: np.ndarray) -> np.ndarray:
    """参考预处理：224 resize_with_pad + JPEG q95，RGB 原样不做 cvtColor。"""
    ok, enc = cv2.imencode(".jpg", resize_with_pad_cv2(img, SIZE, SIZE), JPEG_PARAM)
    if not ok:
        raise ValueError("JPEG 编码失败")
    return enc.reshape(-1).astype(np.uint8)


def decode_jpeg(buf: np.ndarray) -> np.ndarray:
    img = cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("JPEG 解码失败")
    return img


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- 写入逻辑（单测故障注入的改写点） ----------------


def select_views(ts) -> tuple[np.ndarray, np.ndarray]:
    """取一帧的 (前视, 腕部) 原图。"""
    return ts["obs"]["front_rgb"][()], ts["obs"]["wrist_rgb"][()]


def derive_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    """state = joint_state ⊕ gripper_state[:1]，f32。"""
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def derive_action_seq(raw_actions: list[np.ndarray]) -> np.ndarray:
    """action[t] = joint_action[t]，不移位不丢帧，f32 [T,8]。"""
    return np.stack(raw_actions, axis=0).astype(np.float32)


def load_episode_map(records_dir: str, task: str) -> list[dict]:
    rows = json.load(open(pathlib.Path(records_dir) / f"record_dataset_{task}_episode_map.json"))
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("episodes") or list(rows.values())
    return rows


def build_jobs(group: str, extract_root: str, records_dir: str, manifest_path: str, pkl_dir: str, out_dir: str) -> list[dict]:
    """按 episode_map 与 manifest 生成本组的任务列表。"""
    task = group.split("_")[2]
    rows = load_episode_map(records_dir, task)
    manifest = json.load(open(manifest_path))
    by_key = {(e["h5_file"], e["raw_ep_idx"]): e for e in manifest["episodes"]}
    out_dir_p = pathlib.Path(out_dir) / "episodes"
    jobs = []
    for r in rows:
        if not r["member"].startswith(group + "/"):
            continue
        e = by_key[(f"record_dataset_{task}.h5", r["new_idx"])]
        jobs.append(
            {
                "path": str(pathlib.Path(extract_root) / r["member"]),
                "out_path": str(out_dir_p / f"ep_{e['global_episode_idx']:04d}.h5"),
                "pkl_path": str(pathlib.Path(pkl_dir) / f"{e['exec_sample_offset']}.pkl"),
                "records_dir": str(records_dir),
                "manifest_path": str(manifest_path),
                "global": e["global_episode_idx"],
                "task": r["task"],
                "difficulty": r["difficulty"],
                "episode": r["episode"],
                "seed": r["seed"],
                "member": r["member"],
                "h5_sha256": r["h5_sha256"],
                "h5_bytes": r["h5_bytes"],
                "src_group": r["src_group"],
                "num_timesteps": e["num_timesteps"],
                "exec_start_idx": e["exec_start_idx"],
            }
        )
    return jobs


# ---------------- 独立校验（不调用上面的写入函数） ----------------


@functools.lru_cache(maxsize=8)
def _reference_tables(records_dir: str, manifest_path: str, task: str):
    rows = load_episode_map(records_dir, task)
    manifest = json.load(open(manifest_path))
    rows_by_member = {r["member"]: r for r in rows}
    if len(rows_by_member) != len(rows):
        raise ValueError(f"episode_map {task} 有重复 member")
    man_by_key = {(e["h5_file"], e["raw_ep_idx"]): e for e in manifest["episodes"]}
    return rows_by_member, man_by_key


def check_mapping(job: dict) -> dict:
    """由 member 独立查 episode_map → manifest，逐条比对 job 的映射字段；返回 manifest 条目。"""
    task = job["member"].split("/")[0].split("_")[2]
    rows_by_member, man_by_key = _reference_tables(job["records_dir"], job["manifest_path"], task)
    r = rows_by_member.get(job["member"])
    if r is None:
        raise ValueError(f"映射不符：{job['member']} 不在 episode_map")
    for k in ("task", "difficulty", "episode", "seed", "h5_sha256", "h5_bytes", "src_group"):
        if job[k] != r[k]:
            raise ValueError(f"映射不符：{job['member']} {k}={job[k]!r} != episode_map {r[k]!r}")
    e = man_by_key.get((f"record_dataset_{task}.h5", r["new_idx"]))
    if e is None:
        raise ValueError(f"映射不符：{job['member']} new_idx={r['new_idx']} 不在 manifest")
    g = e["global_episode_idx"]
    expect = {
        "global": g,
        "num_timesteps": e["num_timesteps"],
        "exec_start_idx": e["exec_start_idx"],
    }
    for k, v in expect.items():
        if job[k] != v:
            raise ValueError(f"映射不符：{job['member']} {k}={job[k]} != 参考 {v}")
    if r.get("timestep_count", e["num_timesteps"]) != e["num_timesteps"]:
        raise ValueError(f"映射不符：{job['member']} episode_map 帧数 {r['timestep_count']} != manifest {e['num_timesteps']}")
    if pathlib.Path(job["out_path"]).name != f"ep_{g:04d}.h5":
        raise ValueError(f"映射不符：{job['member']} 输出 {job['out_path']} != ep_{g:04d}.h5")
    if pathlib.Path(job["pkl_path"]).name != f"{e['exec_sample_offset']}.pkl":
        raise ValueError(f"映射不符：{job['member']} pkl {job['pkl_path']} != {e['exec_sample_offset']}.pkl")
    return e


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def verify_written(src: pathlib.Path, ep, out_file: h5py.File, job: dict, task_goal: str) -> None:
    """读回临时输出，与原始字段独立重算值逐位比对。"""
    n = job["num_timesteps"]
    ex = job["exec_start_idx"]
    front = out_file["front_jpeg"]
    wrist = out_file["wrist_jpeg"]
    state = out_file["state"][()]
    action = out_file["action"][()]
    is_demo = out_file["is_demo"][()]
    if len(front) != n or len(wrist) != n:
        raise ValueError(f"派生值不符：{src} JPEG 条数 {len(front)}/{len(wrist)} != {n}")
    if state.shape != (n, 8) or state.dtype != np.float32 or action.shape != (n, 8) or action.dtype != np.float32:
        raise ValueError(f"派生值不符：{src} state {state.shape}{state.dtype} action {action.shape}{action.dtype}")
    if is_demo.shape != (n,) or is_demo.dtype != np.bool_:
        raise ValueError(f"派生值不符：{src} is_demo {is_demo.shape}{is_demo.dtype}")
    for t in range(n):
        ts = ep[f"timestep_{t}"]
        obs = ts["obs"]
        js = np.asarray(obs["joint_state"][()], dtype=np.float32)
        gs = np.asarray(obs["gripper_state"][()], dtype=np.float32)
        ref_state = np.array([*js.tolist(), gs[0]], dtype=np.float32)
        if js.shape != (7,) or not _bits_equal(state[t], ref_state):
            raise ValueError(f"派生值不符：{src} t={t} state 写入 {state[t]} != 重算 {ref_state}")
        ref_action = np.asarray(ts["action"]["joint_action"][()]).astype(np.float32)
        if ref_action.shape != (8,) or not _bits_equal(action[t], ref_action):
            raise ValueError(f"派生值不符：{src} t={t} action 写入 {action[t]} != 重算 {ref_action}")
        if bool(is_demo[t]) != bool(ts["info"]["is_video_demo"][()]):
            raise ValueError(f"派生值不符：{src} t={t} is_demo")
        if len(front[t]) < 4 or len(wrist[t]) < 4 or bytes(front[t][:2]) != b"\xff\xd8" or bytes(wrist[t][:2]) != b"\xff\xd8":
            raise ValueError(f"派生值不符：{src} t={t} JPEG 头非法")
    probe = sorted({0, n - 1, ex - 1, ex} & set(range(n)))  # 首帧、尾帧、演示末帧、执行首帧
    for t in probe:
        obs = ep[f"timestep_{t}"]["obs"]
        for name, ds in (("front_rgb", front), ("wrist_rgb", wrist)):
            ref = decode_jpeg(encode_jpeg(obs[name][()]))
            got = decode_jpeg(ds[t])
            if ref.shape != (SIZE, SIZE, 3) or got.shape != ref.shape or not np.array_equal(ref, got):
                raise ValueError(f"派生值不符：{src} t={t} {name} 解码像素与参考预处理不等")
    attrs = out_file.attrs
    expect = {
        "global_episode_idx": job["global"],
        "task": job["task"],
        "difficulty": job["difficulty"],
        "episode": job["episode"],
        "seed": job["seed"],
        "member": job["member"],
        "h5_sha256": job["h5_sha256"],
        "num_timesteps": n,
        "exec_start_idx": ex,
        "task_goal": task_goal,
    }
    for k, v in expect.items():
        got = attrs.get(k)
        if isinstance(got, bytes):
            got = got.decode()
        if got != v:
            raise ValueError(f"派生值不符：{src} attrs[{k}]={got!r} != {v!r}")


def process_one(job: dict) -> dict:
    src = pathlib.Path(job["path"])
    out_path = pathlib.Path(job["out_path"])
    check_mapping(job)
    if out_path.exists():
        with h5py.File(out_path, "r") as f:
            if f.attrs.get("complete", False) and f.attrs.get("member") == job["member"]:
                return {"global": job["global"], "status": "skip", "frames": int(f.attrs["num_timesteps"])}
        out_path.unlink()

    size = src.stat().st_size
    if size != job["h5_bytes"]:
        raise ValueError(f"{src}: 字节数 {size} != episode_map {job['h5_bytes']}")
    digest = sha256_file(src)
    if digest != job["h5_sha256"]:
        raise ValueError(f"{src}: sha256 {digest} != episode_map {job['h5_sha256']}")

    tmp = out_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with h5py.File(src, "r") as f:
            groups = [k for k in f.keys() if k.startswith("episode_")]
            if groups != [job["src_group"]]:
                raise ValueError(f"{src}: 顶层组 {groups} != [{job['src_group']}]")
            ep = f[job["src_group"]]
            task_goal = ep["setup"]["task_goal"][()][0].decode().lower()
            n = sum(1 for k in ep.keys() if k.startswith("timestep_"))
            if n != job["num_timesteps"]:
                raise ValueError(f"{src}: 帧数 {n} != manifest {job['num_timesteps']}")
            ex = job["exec_start_idx"]

            # 写入
            front_j = np.empty(n, dtype=object)
            wrist_j = np.empty(n, dtype=object)
            states = np.zeros((n, 8), dtype=np.float32)
            raw_actions = []
            is_demo = np.zeros(n, dtype=bool)
            for t in range(n):
                ts = ep[f"timestep_{t}"]
                front, wrist = select_views(ts)
                for nm, img in (("front", front), ("wrist", wrist)):
                    if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                        raise ValueError(f"{src}: t={t} {nm} {img.shape} {img.dtype}")
                front_j[t] = encode_jpeg(front)
                wrist_j[t] = encode_jpeg(wrist)
                states[t] = derive_state(ts["obs"]["joint_state"][()], ts["obs"]["gripper_state"][()])
                raw_actions.append(ts["action"]["joint_action"][()])
                is_demo[t] = bool(ts["info"]["is_video_demo"][()])
            actions = derive_action_seq(raw_actions)

            # 校验 2：演示前缀
            demo_len = int(np.argmin(is_demo)) if not is_demo.all() else n
            if demo_len != ex or is_demo[demo_len:].any():
                raise ValueError(f"{src}: demo 前缀 {demo_len} != exec_start_idx {ex}")

            # 校验 3：执行首帧对 V17 pkl（原始字段直接读，不经写入函数）
            with open(job["pkl_path"], "rb") as pf:
                sample = pickle.load(pf)
            if int(sample["epis_idx"][0]) != job["global"] or int(sample["step_idx"][0]) != ex:
                raise ValueError(f"{src}: pkl 指向 epis={sample['epis_idx']} step={sample['step_idx']}")
            obs_ex = ep[f"timestep_{ex}"]["obs"]
            if not np.array_equal(sample["image"], obs_ex["front_rgb"][()]):
                raise ValueError(f"{src}: 执行首帧 front_rgb 与 V17 pkl image 不一致")
            if not np.array_equal(sample["wrist_image"], obs_ex["wrist_rgb"][()]):
                raise ValueError(f"{src}: 执行首帧 wrist_rgb 与 V17 pkl wrist_image 不一致")
            raw_state = np.concatenate([obs_ex["joint_state"][()], obs_ex["gripper_state"][()][:1]], axis=0, dtype=np.float32)
            if not _bits_equal(sample["state"], raw_state):
                raise ValueError(f"{src}: 执行首帧原始 state 与 V17 pkl state 不一致（原始 {raw_state}，pkl {sample['state']}）")

            with h5py.File(tmp, "w") as o:
                o.create_dataset("front_jpeg", data=front_j, dtype=h5py.vlen_dtype(np.uint8))
                o.create_dataset("wrist_jpeg", data=wrist_j, dtype=h5py.vlen_dtype(np.uint8))
                o.create_dataset("state", data=states)
                o.create_dataset("action", data=actions)
                o.create_dataset("is_demo", data=is_demo)
                for k in ("global", "task", "difficulty", "episode", "seed", "member", "h5_sha256", "exec_start_idx"):
                    o.attrs[k if k != "global" else "global_episode_idx"] = job[k]
                o.attrs["num_timesteps"] = n
                o.attrs["task_goal"] = task_goal

            # 校验 4：读回比对，全过才写 complete
            with h5py.File(tmp, "r+") as o:
                verify_written(src, ep, o, job, task_goal)
                if not _bits_equal(o["state"][ex], sample["state"]):
                    raise ValueError(f"派生值不符：{src} 写入 state[{ex}] 与 V17 pkl state 不一致")
                o.attrs["complete"] = True
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"global": job["global"], "status": "ok", "frames": n}


def check_group_jobs(group: str, jobs: list[dict]) -> None:
    """组级映射检查：global 不重复，且与独立参考映射逐条一致。"""
    if not jobs:
        sys.exit(f"拒绝：{group} 在 episode_map 里没有成员")
    globals_ = [j["global"] for j in jobs]
    if len(set(globals_)) != len(globals_):
        raise ValueError(f"映射不符：{group} global 有重复")
    outs = [j["out_path"] for j in jobs]
    if len(set(outs)) != len(outs):
        raise ValueError(f"映射不符：{group} 输出路径有重复")
    for j in jobs:
        check_mapping(j)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", required=True, help="解压根目录（含 record_dataset_<Task>_<diff>/hdf5_files）")
    ap.add_argument("--group", required=True, help="本次处理的归档名，如 record_dataset_BinFill_easy")
    ap.add_argument("--records-dir", required=True, help="policy 仓 docs/dataset-build-doc/4task-v2-1600ep-604f16da/records")
    ap.add_argument("--episode-manifest", required=True)
    ap.add_argument("--pkl-dir", required=True, help="V17 训练库 source/data")
    ap.add_argument("--out", default="/scratch/hongze/NativeMEM/data/robomme4task-v2")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--stream", action="store_true", help="从 stdin 读 tar 流，边解边处理边删")
    args = ap.parse_args(argv)

    (pathlib.Path(args.out) / "episodes").mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.group, args.extract_root, args.records_dir, args.episode_manifest, args.pkl_dir, args.out)
    check_group_jobs(args.group, jobs)
    print(f"GROUP={args.group} episodes={len(jobs)}", flush=True)
    done = frames = 0

    def report(res):
        nonlocal done, frames
        done += 1
        frames += res["frames"]
        if done % 20 == 0 or done == len(jobs):
            print(f"进度 {args.group} {done}/{len(jobs)} frames={frames}", flush=True)

    if not args.stream:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, j) for j in jobs]
            for fu in as_completed(futs):
                report(fu.result())
    else:
        # 流式：从 stdin 读 tar 流（上游 xz -dc），逐成员落盘 → 处理 → 删除，磁盘峰值只有几个成员
        import tarfile

        by_member = {j["member"]: j for j in jobs}
        seen = set()
        root = pathlib.Path(args.extract_root)

        def finish(fu, path):
            try:
                report(fu.result())
            finally:
                pathlib.Path(path).unlink(missing_ok=True)

        pending = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex, tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                j = by_member.get(m.name)
                if j is None:
                    print(f"跳过不在 episode_map 的成员 {m.name}", flush=True)
                    continue
                if m.name in seen:
                    raise ValueError(f"归档里成员重复 {m.name}")
                dst = root / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as fsrc, open(dst, "wb") as fdst:
                    while True:
                        buf = fsrc.read(16 << 20)
                        if not buf:
                            break
                        fdst.write(buf)
                seen.add(m.name)
                pending.append((ex.submit(process_one, j), str(dst)))
                while len(pending) >= args.workers * 2:
                    fu, p = pending.pop(0)
                    finish(fu, p)
            for fu, p in pending:
                finish(fu, p)
        missing = sorted(set(by_member) - seen)
        if missing:
            sys.exit(f"拒绝：归档里缺 {len(missing)} 个 episode_map 成员，首个 {missing[0]}")
    if done != len(jobs):
        sys.exit(f"拒绝：{args.group} 完成 {done} != {len(jobs)}")
    print(f"DATA_V2_GROUP=PASS group={args.group} episodes={done} frames={frames}", flush=True)


if __name__ == "__main__":
    main()
