"""从 4task-v2-1600ep 原始单 episode H5 抽取记忆帧（Stage 1.5 的输入）。

对每个 episode 的每一帧前视图 front_rgb（256×256 RGB）做与 NativeMEM Stage 1 完全相同的预处理：
resize_with_pad_cv2 到 224×224，再 cv2.imencode JPEG（quality 95）。通道顺序保持 RGB 原样
（RMBench 的 pkl2hdf5 也是对 RGB 数组直接 imencode，解码回来仍是 RGB）。

输出：<out>/episodes/ep_<global_episode_idx:04d>.h5，内含
  front_jpeg  vlen uint8 [T]   每帧一个 JPEG 字节串
  is_demo     bool [T]         info/is_video_demo
  attrs: global_episode_idx, task, difficulty, episode, seed, member, h5_sha256, num_timesteps, exec_start_idx, task_goal
每个 episode 同时做三项校验：
  1. 原始 H5 的 sha256 == episode_map.h5_sha256、字节数 == h5_bytes；
  2. 帧数 == episode_manifest.num_timesteps，demo 前缀长度 == exec_start_idx；
  3. 第一个执行帧的原始 front_rgb 与 V17 训练库 source/data/<exec_sample_offset>.pkl 的 image 逐字节相等（证明帧号映射正确）。
任何一项不符即抛错，不写输出。
"""

from __future__ import annotations

import argparse
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


def resize_with_pad_cv2(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """与 NativeMEM scripts/prepare_mem_tokenizer_h5.py::resize_with_pad_cv2 逐行一致。"""
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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_one(job: dict) -> dict:
    src = pathlib.Path(job["path"])
    out_path = pathlib.Path(job["out_path"])
    if out_path.exists():
        with h5py.File(out_path, "r") as f:
            if f.attrs.get("complete", False):
                return {"global": job["global"], "status": "skip", "frames": int(f.attrs["num_timesteps"])}
        out_path.unlink()

    size = src.stat().st_size
    if size != job["h5_bytes"]:
        raise ValueError(f"{src}: 字节数 {size} != episode_map {job['h5_bytes']}")
    digest = sha256_file(src)
    if digest != job["h5_sha256"]:
        raise ValueError(f"{src}: sha256 {digest} != episode_map {job['h5_sha256']}")

    with h5py.File(src, "r") as f:
        groups = [k for k in f.keys() if k.startswith("episode_")]
        if groups != [job["src_group"]]:
            raise ValueError(f"{src}: 顶层组 {groups} != [{job['src_group']}]")
        ep = f[job["src_group"]]
        task_goal = ep["setup"]["task_goal"][()][0].decode()
        n = sum(1 for k in ep.keys() if k.startswith("timestep_"))
        if n != job["num_timesteps"]:
            raise ValueError(f"{src}: 帧数 {n} != manifest {job['num_timesteps']}")
        jpegs = np.empty(n, dtype=object)
        is_demo = np.zeros(n, dtype=bool)
        param = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
        first_exec_raw = None
        for t in range(n):
            ts = ep[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise ValueError(f"{src}: t={t} front_rgb {img.shape} {img.dtype}")
            is_demo[t] = bool(ts["info"]["is_video_demo"][()])
            ok, enc = cv2.imencode(".jpg", resize_with_pad_cv2(img, SIZE, SIZE), param)
            if not ok:
                raise ValueError(f"{src}: t={t} JPEG 编码失败")
            jpegs[t] = enc.reshape(-1).astype(np.uint8)
            if t == job["exec_start_idx"]:
                first_exec_raw = img
    demo_len = int(np.argmin(is_demo)) if not is_demo.all() else n
    if demo_len != job["exec_start_idx"] or is_demo[demo_len:].any():
        raise ValueError(f"{src}: demo 前缀 {demo_len} != exec_start_idx {job['exec_start_idx']}")
    with open(job["pkl_path"], "rb") as pf:
        sample = pickle.load(pf)
    if int(sample["epis_idx"][0]) != job["global"] or int(sample["step_idx"][0]) != job["exec_start_idx"]:
        raise ValueError(f"{src}: pkl 指向 epis={sample['epis_idx']} step={sample['step_idx']}")
    if not np.array_equal(sample["image"], first_exec_raw):
        raise ValueError(f"{src}: 第一个执行帧与 V17 pkl image 不一致")

    tmp = out_path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as o:
        o.create_dataset("front_jpeg", data=jpegs, dtype=h5py.vlen_dtype(np.uint8))
        o.create_dataset("is_demo", data=is_demo)
        for k in ("global", "task", "difficulty", "episode", "seed", "member", "h5_sha256", "exec_start_idx"):
            o.attrs[k if k != "global" else "global_episode_idx"] = job[k]
        o.attrs["num_timesteps"] = n
        o.attrs["task_goal"] = task_goal
        o.attrs["complete"] = True
    os.replace(tmp, out_path)
    return {"global": job["global"], "status": "ok", "frames": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", required=True, help="解压后的根目录（含 record_dataset_<Task>_<diff>/hdf5_files）")
    ap.add_argument("--group", required=True, help="本次处理的归档名，如 record_dataset_BinFill_easy")
    ap.add_argument("--records-dir", required=True, help="policy 仓 docs/dataset-build-doc/4task-v2-1600ep-604f16da/records")
    ap.add_argument("--episode-manifest", required=True)
    ap.add_argument("--pkl-dir", required=True, help="V17 训练库 source/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--stream", action="store_true", help="从 stdin 读 tar 流，边解边处理边删")
    args = ap.parse_args()

    manifest = json.load(open(args.episode_manifest))
    by_key = {(e["h5_file"], e["raw_ep_idx"]): e for e in manifest["episodes"]}
    out_dir = pathlib.Path(args.out) / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    task = args.group.split("_")[2]
    rows = json.load(open(pathlib.Path(args.records_dir) / f"record_dataset_{task}_episode_map.json"))
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("episodes") or list(rows.values())
    jobs = []
    for r in rows:
        if not r["member"].startswith(args.group + "/"):
            continue
        e = by_key[(f"record_dataset_{task}.h5", r["new_idx"])]
        jobs.append(
            {
                "path": str(pathlib.Path(args.extract_root) / r["member"]),
                "out_path": str(out_dir / f"ep_{e['global_episode_idx']:04d}.h5"),
                "pkl_path": str(pathlib.Path(args.pkl_dir) / f"{e['exec_sample_offset']}.pkl"),
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
    if not jobs:
        sys.exit(f"拒绝：{args.group} 在 episode_map 里没有成员")
    print(f"GROUP={args.group} episodes={len(jobs)}", flush=True)
    done = frames = 0
    if not args.stream:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, j) for j in jobs]
            for fu in as_completed(futs):
                res = fu.result()
                done += 1
                frames += res["frames"]
                if done % 20 == 0 or done == len(jobs):
                    print(f"进度 {args.group} {done}/{len(jobs)} frames={frames}", flush=True)
    else:
        # 流式：从 stdin 读 tar 流（上游 xz -dc），逐成员落盘 → 处理 → 删除，磁盘峰值只有几个成员
        import tarfile

        by_member = {j["member"]: j for j in jobs}
        seen = set()
        root = pathlib.Path(args.extract_root)

        def finish(fu, path):
            nonlocal done, frames
            res = fu.result()
            pathlib.Path(path).unlink(missing_ok=True)
            done += 1
            frames += res["frames"]
            if done % 20 == 0 or done == len(jobs):
                print(f"进度 {args.group} {done}/{len(jobs)} frames={frames}", flush=True)

        pending = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex, tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                j = by_member.get(m.name)
                if j is None:
                    print(f"跳过不在 episode_map 的成员 {m.name}", flush=True)
                    continue
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
    print(f"MEMFRAMES_GROUP=PASS group={args.group} episodes={done} frames={frames}", flush=True)


if __name__ == "__main__":
    main()
