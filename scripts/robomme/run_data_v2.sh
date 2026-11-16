#!/usr/bin/env bash
# 重抽 NativeMEM Stage 2 训练数据（第二版：前视＋腕部 JPEG、state、action、is_demo）。
# 阶段0 先跑故障注入单测取 faults；阶段1 下载 14 个主归档（raw-dl 里带 .ok 的直接复用）；
# 阶段2 14 组并行 xz -dc -T0 | build_data_v2.py --stream，组完成即删归档与解压目录；阶段3 汇总校验。
# 用法：bash scripts/robomme/run_data_v2.sh   （放 detached tmux 里跑）
set -euo pipefail
N=/scratch/hongze/NativeMEM
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
D=$P/v1-store/datasets/4task-v2-1600ep-604f16da
REC=$P/docs/dataset-build-doc/4task-v2-1600ep-604f16da/records
BUCKET=hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2
DL=$N/data/raw-dl
X=$N/data/raw-x2
OUT=$N/data/robomme4task-v2
LOGD=$N/logs/data_v2
PY="$N/.venv/bin/python"
mkdir -p "$DL" "$X" "$OUT" "$LOGD"
export HF_HOME=/scratch/hongze/.cache/huggingface UV_CACHE_DIR=/scratch/hongze/.cache/uv XDG_CACHE_HOME=/scratch/hongze/.cache

echo "阶段0：故障注入单测 $(date -u +%FT%TZ)"
FJ=$LOGD/faults.json
rm -f "$FJ"
( cd "$N" && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= DATA_V2_FAULTS_JSON="$FJ" \
    "$PY" -m pytest scripts/robomme/test_build_data_v2.py -q -s -p no:cacheprovider ) > "$LOGD/unittest.log" 2>&1 \
  || { tail -20 "$LOGD/unittest.log"; echo "DATA_V2=FAIL 单测失败，见 $LOGD/unittest.log"; exit 1; }
FAULTS=$("$PY" -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d[\"caught\"]}/{d[\"total\"]}")
' "$FJ")
[ "$FAULTS" = "4/4" ] || { echo "DATA_V2=FAIL 故障注入只拦下 $FAULTS"; exit 1; }
grep -E "REAL_EPISODE|passed|skipped" "$LOGD/unittest.log" || true
echo "阶段0完成 faults=$FAULTS"

set -a; . "$P/v1-store/secrets/hf.env"; set +a
HF=(uv run --no-project --quiet --with 'huggingface_hub[cli]>=1.0' hf)
cd "$N/logs"   # 不在任一项目根下运行 uv --no-project 的 hf

GROUPS_ALL=(
  BinFill_easy BinFill_medium BinFill_hard
  RouteStick_easy RouteStick_medium RouteStick_hard RouteStick_xhard
  VideoRepick_easy VideoRepick_medium VideoRepick_xhard
  VideoUnmaskSwap_easy VideoUnmaskSwap_medium VideoUnmaskSwap_hard VideoUnmaskSwap_xhard
)

echo "阶段1：下载 $(date -u +%FT%TZ)"
dl_one() {
  local g=$1 f="record_dataset_$1.h5.tar.xz"
  if [ -f "$LOGD/$g.done" ]; then echo "已处理过 $g，跳过下载"; return 0; fi
  if [ -f "$DL/$f" ] && [ -f "$DL/$f.ok" ]; then echo "复用已下载 $f"; return 0; fi
  rm -f "$DL/$f"
  "${HF[@]}" buckets cp "$BUCKET/$f" "$DL/$f" > "$LOGD/dl_$g.log" 2>&1
  touch "$DL/$f.ok"; echo "下载完成 $f $(stat -c %s "$DL/$f")"
}
for g in "${GROUPS_ALL[@]}"; do
  dl_one "$g" &
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n || true; done
done
wait || true
# 成败以落盘标记为准（wait -n 回收过的进程无法再取退出码）
dlfail=0
for g in "${GROUPS_ALL[@]}"; do
  f="record_dataset_$g.h5.tar.xz"
  if [ -f "$LOGD/$g.done" ] || { [ -f "$DL/$f" ] && [ -f "$DL/$f.ok" ]; }; then :; else
    echo "下载失败 $g，见 $LOGD/dl_$g.log"; dlfail=1
  fi
done
[ "$dlfail" = 0 ] || { echo "DATA_V2=FAIL 下载失败"; exit 1; }
echo "阶段1完成 $(date -u +%FT%TZ)"

echo "阶段2：流式解压重抽（14 组并行，每组 6 进程）"
run_group() {
  local g=$1 f="$DL/record_dataset_$1.h5.tar.xz"
  if [ -f "$LOGD/$g.done" ]; then echo "已完成 $g"; return 0; fi
  set -o pipefail
  xz -dc -T0 "$f" | "$PY" -I "$N/scripts/robomme/build_data_v2.py" --stream \
      --extract-root "$X/$g" --group "record_dataset_$g" --records-dir "$REC" \
      --episode-manifest "$D/meta/episode_manifest.json" --pkl-dir "$D/source/data" \
      --out "$OUT" --workers 6 > "$LOGD/$g.log" 2>&1
  grep -q "^DATA_V2_GROUP=PASS" "$LOGD/$g.log"
  touch "$LOGD/$g.done"; rm -f "$f" "$f.ok"; rm -rf "${X:?}/$g"
  tail -1 "$LOGD/$g.log"
}
fail=0; pids=(); names=()
for g in "${GROUPS_ALL[@]}"; do run_group "$g" & pids+=($!); names+=("$g"); done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "失败 ${names[$i]}，见 $LOGD/${names[$i]}.log"; tail -3 "$LOGD/${names[$i]}.log"; fail=1; fi
done
[ "$fail" = 0 ] || { echo "DATA_V2=FAIL 有组失败"; exit 1; }

echo "阶段3：汇总校验 $(date -u +%FT%TZ)"
"$PY" -I - "$OUT" "$D/meta/episode_manifest.json" "$REC" "$LOGD" "$FAULTS" <<'EOF'
import sys, json, pathlib, h5py
out = pathlib.Path(sys.argv[1]) / "episodes"
man = json.load(open(sys.argv[2]))
rec, logd, faults = pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5]
man_by_key = {(e["h5_file"], e["raw_ep_idx"]): e for e in man["episodes"]}
rows = {}
for task in ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"):
    for r in json.load(open(rec / f"record_dataset_{task}_episode_map.json")):
        rows[r["member"]] = r
bad, frames, seen = [], 0, set()
for e in man["episodes"]:
    g = e["global_episode_idx"]
    p = out / f"ep_{g:04d}.h5"
    try:
        with h5py.File(p, "r") as f:
            a = f.attrs
            n = int(a["num_timesteps"])
            r = rows[a["member"]]
            ref = man_by_key[(f"record_dataset_{r['task']}.h5", r["new_idx"])]
            ok = (
                bool(a["complete"]) and n == e["num_timesteps"] and int(a["global_episode_idx"]) == g
                and ref["global_episode_idx"] == g and a["h5_sha256"] == r["h5_sha256"]
                and int(a["exec_start_idx"]) == e["exec_start_idx"]
                and len(f["front_jpeg"]) == n and len(f["wrist_jpeg"]) == n
                and f["state"].shape == (n, 8) and f["action"].shape == (n, 8) and f["is_demo"].shape == (n,)
            )
            seen.add(a["member"])
            frames += n
    except Exception as ex:
        ok = False
    if not ok:
        bad.append(g)
glogs = [p for p in sorted(logd.glob("*_*.log")) if not p.name.startswith(("dl_", "unittest"))]
groups_ok = len(glogs) == 14 and all(
    any(l.startswith("DATA_V2_GROUP=PASS") for l in open(p, errors="replace")) for p in glogs
)
tot = man["totals"]["timesteps"]
n_ok = len(man["episodes"]) - len(bad)
derived = "PASS" if not bad and groups_ok and len(seen) == len(man["episodes"]) else "FAIL"
st = "PASS" if derived == "PASS" and n_ok == 1600 and frames == tot == 1192918 and faults == "4/4" else "FAIL"
print(f"DATA_V2={st} episodes={n_ok} frames={frames} views=2 derived={derived} faults={faults}" + ("" if st == "PASS" else f" expect_frames={tot} bad={bad[:5]} groups_ok={groups_ok}"))
sys.exit(0 if st == "PASS" else 1)
EOF
