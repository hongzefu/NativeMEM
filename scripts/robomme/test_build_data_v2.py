"""build_data_v2.py 的 CPU 单测：合成正向、流式整组正向、4 种故障注入必须被拦、真实单集只读正向。

4 种故障注入的结果写成 json（路径取环境变量 DATA_V2_FAULTS_JSON，缺省为系统临时目录下 data_v2_faults.json），
run_data_v2.sh 开头读取它得到 faults=<caught>/<total>。
故障注入一律用 monkeypatch 改写 build_data_v2 的写入函数，生产路径没有任何故障开关。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time

import h5py
import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_data_v2 as bd  # noqa: E402

TASK = "BinFill"
GROUP = f"record_dataset_{TASK}_easy"
POLICY = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
REAL_RECORDS = POLICY / "docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"
REAL_DS = POLICY / "v1-store/datasets/4task-v2-1600ep-604f16da"
RAW_DL = pathlib.Path("/scratch/hongze/NativeMEM/data/raw-dl")
FAULTS_JSON = pathlib.Path(os.environ.get("DATA_V2_FAULTS_JSON", pathlib.Path(tempfile.gettempdir()) / "data_v2_faults.json"))
FAULT_NAMES = ("wrist_as_front", "action_shift_t1", "gripper_wrong_col", "swap_global")


# ---------------- 合成数据 ----------------


def _write_episode_h5(path: pathlib.Path, rng: np.random.Generator, n: int, ex: int, goal: str) -> dict:
    """按 RoboMME 单集 H5 的 schema 写一个合成 episode，返回执行首帧原始值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = {}
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_0")
        ep.create_group("setup").create_dataset("task_goal", data=np.array([goal.encode(), b"alt goal"], dtype=object), dtype=h5py.string_dtype())
        for t in range(n):
            ts = ep.create_group(f"timestep_{t}")
            obs = ts.create_group("obs")
            front = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            js = rng.standard_normal(7).astype(np.float32)
            gs = np.array([rng.uniform(0, 0.04), rng.uniform(0.05, 0.08)], dtype=np.float32)  # 两列必不同
            obs.create_dataset("front_rgb", data=front)
            obs.create_dataset("wrist_rgb", data=wrist)
            obs.create_dataset("joint_state", data=js)
            obs.create_dataset("gripper_state", data=gs)
            ts.create_group("info").create_dataset("is_video_demo", data=np.bool_(t < ex))
            ts.create_group("action").create_dataset("joint_action", data=rng.standard_normal(8).astype(np.float64))
            if t == ex:
                first = {"image": front, "wrist_image": wrist, "state": np.concatenate([js, gs[:1]]).astype(np.float32)}
    return first


@pytest.fixture()
def synth(tmp_path):
    """两集合成数据：extract_root、episode_map、manifest、pkl 目录、输出目录。"""
    rng = np.random.default_rng(0)
    root = tmp_path / "x"
    rec = tmp_path / "records"
    pkl = tmp_path / "pkl"
    out = tmp_path / "out"
    for d in (root, rec, pkl, out):
        d.mkdir()
    specs = [(0, 7, 3, 0), (1, 6, 2, 4)]  # (global, T, exec_start, exec_sample_offset)
    rows, eps = [], []
    for g, n, ex, off in specs:
        member = f"{GROUP}/hdf5_files/{TASK}_ep{g}_seed{4000 + g}.h5"
        p = root / member
        first = _write_episode_h5(p, rng, n, ex, f"Put Cube {g} INTO bin")
        rows.append(
            {
                "task": TASK, "difficulty": "easy", "episode": g, "seed": 4000 + g, "role": "primary",
                "member": member, "h5_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "h5_bytes": p.stat().st_size,
                "timestep_count": n, "new_idx": g, "src_group": "episode_0",
            }
        )
        eps.append(
            {
                "global_episode_idx": g, "h5_file": f"record_dataset_{TASK}.h5", "raw_ep_idx": g, "num_timesteps": n,
                "exec_start_idx": ex, "exec_samples": n - ex, "exec_sample_offset": off, "total_sample_offset": 0,
            }
        )
        sample = dict(first)
        sample["epis_idx"] = np.array([g], dtype=np.int32)
        sample["step_idx"] = np.array([ex], dtype=np.int32)
        with open(pkl / f"{off}.pkl", "wb") as f:
            pickle.dump(sample, f)
    (rec / f"record_dataset_{TASK}_episode_map.json").write_text(json.dumps(rows))
    man = tmp_path / "episode_manifest.json"
    man.write_text(json.dumps({"episodes": eps, "totals": {"episodes": 2, "timesteps": 13}}))
    bd._reference_tables.cache_clear()
    return {"root": root, "rec": rec, "pkl": pkl, "out": out, "man": man, "rows": rows, "specs": specs}


def _jobs(s):
    return bd.build_jobs(GROUP, str(s["root"]), str(s["rec"]), str(s["man"]), str(s["pkl"]), str(s["out"]))


def _run_group(s):
    (s["out"] / "episodes").mkdir(parents=True, exist_ok=True)
    jobs = bd.build_jobs(GROUP, str(s["root"]), str(s["rec"]), str(s["man"]), str(s["pkl"]), str(s["out"]))
    bd.check_group_jobs(GROUP, jobs)
    return [bd.process_one(j) for j in jobs]


def _no_complete_output(s) -> bool:
    eps = s["out"] / "episodes"
    return not eps.exists() or not any(eps.iterdir())


# ---------------- 正向 ----------------


def test_synthetic_positive(synth):
    res = _run_group(synth)
    assert [r["status"] for r in res] == ["ok", "ok"]
    for g, n, ex, _ in synth["specs"]:
        p = synth["out"] / "episodes" / f"ep_{g:04d}.h5"
        src = synth["root"] / synth["rows"][g]["member"]
        with h5py.File(p, "r") as o, h5py.File(src, "r") as f:
            ep = f["episode_0"]
            assert o.attrs["complete"] and o.attrs["num_timesteps"] == n and o.attrs["exec_start_idx"] == ex
            assert o.attrs["global_episode_idx"] == g and o.attrs["member"] == synth["rows"][g]["member"]
            assert o.attrs["task_goal"] == f"put cube {g} into bin"
            assert o["state"].shape == (n, 8) and o["state"].dtype == np.float32
            assert o["action"].shape == (n, 8) and o["action"].dtype == np.float32
            assert o["is_demo"][()].tolist() == [t < ex for t in range(n)]
            for t in range(n):
                ts = ep[f"timestep_{t}"]
                exp = np.concatenate([ts["obs/joint_state"][()], ts["obs/gripper_state"][()][:1]]).astype(np.float32)
                assert np.array_equal(o["state"][t], exp)
                assert np.array_equal(o["action"][t], ts["action/joint_action"][()].astype(np.float32))
                for name, key in (("front_jpeg", "obs/front_rgb"), ("wrist_jpeg", "obs/wrist_rgb")):
                    assert np.array_equal(bytes(o[name][t]), bytes(bd.encode_jpeg(ts[key][()])))
    # 重跑：已 complete 的直接跳过
    assert [r["status"] for r in _run_group(synth)] == ["skip", "skip"]


def test_stream_main_positive(synth, tmp_path):
    """生产入口 --stream 走进程池整组跑通，并打印组判定行。"""
    tar_path = tmp_path / "g.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(synth["root"] / GROUP, arcname=GROUP)
    xroot = tmp_path / "xs"
    cmd = [
        sys.executable, "-I", str(HERE / "build_data_v2.py"), "--stream", "--extract-root", str(xroot), "--group", GROUP,
        "--records-dir", str(synth["rec"]), "--episode-manifest", str(synth["man"]), "--pkl-dir", str(synth["pkl"]),
        "--out", str(synth["out"]), "--workers", "2",
    ]
    with open(tar_path, "rb") as fin:
        r = subprocess.run(cmd, stdin=fin, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"DATA_V2_GROUP=PASS group={GROUP} episodes=2 frames=13" in r.stdout
    assert not any(p.is_file() for p in xroot.rglob("*.h5"))  # 解压出的成员已删


# ---------------- 故障注入 ----------------


@pytest.fixture(scope="module")
def fault_log():
    log: dict[str, str] = {}
    yield log
    caught = sum(1 for k in FAULT_NAMES if log.get(k) == "caught")
    FAULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FAULTS_JSON.write_text(json.dumps({"faults": log, "caught": caught, "total": len(FAULT_NAMES), "written_at": time.time()}, ensure_ascii=False))


def _expect_caught(fault_log, name, s, pattern):
    try:
        _run_group(s)
    except ValueError as e:
        assert pattern in str(e), f"{name} 被拦但原因不符：{e}"
        assert _no_complete_output(s), f"{name} 被拦后仍留下输出"
        fault_log[name] = "caught"
        return
    fault_log[name] = "missed"
    pytest.fail(f"故障 {name} 未被校验拦下")


def test_fault_wrist_as_front(synth, fault_log, monkeypatch):
    monkeypatch.setattr(bd, "select_views", lambda ts: (ts["obs"]["front_rgb"][()], ts["obs"]["front_rgb"][()]))
    _expect_caught(fault_log, "wrist_as_front", synth, "wrist_rgb 解码像素与参考预处理不等")


def test_fault_action_shift(synth, fault_log, monkeypatch):
    def shifted(raw):
        n = len(raw)
        return np.stack([raw[min(t + 1, n - 1)] for t in range(n)]).astype(np.float32)

    monkeypatch.setattr(bd, "derive_action_seq", shifted)
    _expect_caught(fault_log, "action_shift_t1", synth, "action 写入")


def test_fault_gripper_wrong_col(synth, fault_log, monkeypatch):
    monkeypatch.setattr(bd, "derive_state", lambda js, gs: np.concatenate([js, gs[1:2]], axis=0, dtype=np.float32))
    _expect_caught(fault_log, "gripper_wrong_col", synth, "state 写入")


def test_fault_swap_global(synth, fault_log, monkeypatch):
    orig = bd.build_jobs

    def swapped(*a, **k):
        jobs = orig(*a, **k)
        a0, a1 = jobs[0], jobs[1]
        for key in ("global", "out_path", "pkl_path"):
            a0[key], a1[key] = a1[key], a0[key]
        return jobs

    monkeypatch.setattr(bd, "build_jobs", swapped)
    _expect_caught(fault_log, "swap_global", synth, "映射不符")


# ---------------- 真实单集只读正向 ----------------


def _extract_first_member(dst_dir: pathlib.Path, limit_s: float = 180.0):
    oks = sorted((p for p in RAW_DL.glob("record_dataset_*.h5.tar.xz") if (p.parent / (p.name + ".ok")).exists()), key=lambda p: p.stat().st_size)
    if not oks:
        return None, "raw-dl 下没有带 .ok 的归档"
    arc = oks[0]
    t0 = time.time()
    proc = subprocess.Popen(["xz", "-dc", "-T0", str(arc)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                dst = dst_dir / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as s, open(dst, "wb") as d:
                    while True:
                        buf = s.read(16 << 20)
                        if not buf:
                            break
                        d.write(buf)
                        if time.time() - t0 > limit_s:
                            return None, f"抽取超过 {limit_s:.0f}s"
                return m.name, f"{arc.name} 首成员 {m.name} 用时 {time.time() - t0:.1f}s"
    except (tarfile.TarError, OSError) as e:
        return None, f"抽取失败 {e!r}"
    finally:
        proc.kill()
        proc.wait()
    return None, "归档里没有文件成员"


@pytest.mark.skipif(not REAL_RECORDS.is_dir() or not (REAL_DS / "meta/episode_manifest.json").is_file(), reason="policy 仓 records／manifest 不在")
def test_real_episode_positive(tmp_path):
    xroot = tmp_path / "x"
    out = tmp_path / "out"
    member, msg = _extract_first_member(xroot)
    print(f"REAL_EPISODE {msg}")
    if member is None:
        pytest.skip(msg)
    try:
        group = member.split("/")[0]
        bd._reference_tables.cache_clear()
        (out / "episodes").mkdir(parents=True)
        jobs = bd.build_jobs(group, str(xroot), str(REAL_RECORDS), str(REAL_DS / "meta/episode_manifest.json"), str(REAL_DS / "source/data"), str(out))
        bd.check_group_jobs(group, jobs)
        job = next(j for j in jobs if j["member"] == member)
        res = bd.process_one(job)
        assert res["status"] == "ok" and res["frames"] == job["num_timesteps"]
        with h5py.File(job["out_path"], "r") as o:
            assert o.attrs["complete"] and len(o["front_jpeg"]) == len(o["wrist_jpeg"]) == job["num_timesteps"]
            assert o.attrs["task_goal"] == o.attrs["task_goal"].lower()
            print(f"REAL_EPISODE=PASS global={job['global']} frames={res['frames']} exec={job['exec_start_idx']} goal={o.attrs['task_goal']!r}")
    finally:
        for p in sorted(tmp_path.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
