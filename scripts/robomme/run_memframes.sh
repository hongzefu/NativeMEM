#!/usr/bin/env bash
# 下载 4task-v2-1600ep 的 14 个主归档（HF 私有 bucket），流式解压并抽取记忆帧，最后汇总校验。
# 用法：bash scripts/robomme/run_memframes.sh   （放 detached tmux 里跑）
set -euo pipefail
N=/scratch/hongze/NativeMEM
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
D=$P/v1-store/datasets/4task-v2-1600ep-604f16da
REC=$P/docs/dataset-build-doc/4task-v2-1600ep-604f16da/records
BUCKET=hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2
DL=$N/data/raw-dl
X=$N/data/raw-x
OUT=$N/data/robomme4task-memframes
LOGD=$N/logs/memframes
mkdir -p "$DL" "$X" "$OUT" "$LOGD"
export HF_HOME=/scratch/hongze/.cache/huggingface UV_CACHE_DIR=/scratch/hongze/.cache/uv
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
  if [ -f "$DL/$f" ] && [ -f "$DL/$f.ok" ]; then echo "已下载 $f"; return 0; fi
  "${HF[@]}" buckets cp "$BUCKET/$f" "$DL/$f" > "$LOGD/dl_$g.log" 2>&1
  touch "$DL/$f.ok"; echo "下载完成 $f $(stat -c %s "$DL/$f")"
}
export -f dl_one; export DL LOGD BUCKET; export HF_STR="${HF[*]}"
pids=()
for g in "${GROUPS_ALL[@]}"; do
  dl_one "$g" & pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "阶段1完成 $(date -u +%FT%TZ)"

echo "阶段2：流式解压抽帧（14 组并行，每组 6 进程）"
run_group() {
  local g=$1 f="$DL/record_dataset_$1.h5.tar.xz"
  if [ -f "$LOGD/$g.done" ]; then echo "已完成 $g"; return 0; fi
  set -o pipefail
  xz -dc -T0 "$f" | "$N/.venv/bin/python" -I "$N/scripts/robomme/build_memframes.py" --stream \
      --extract-root "$X/$g" --group "record_dataset_$g" --records-dir "$REC" \
      --episode-manifest "$D/meta/episode_manifest.json" --pkl-dir "$D/source/data" \
      --out "$OUT" --workers 6 > "$LOGD/$g.log" 2>&1
  grep -q "^MEMFRAMES_GROUP=PASS" "$LOGD/$g.log"
  touch "$LOGD/$g.done"; rm -f "$f" "$f.ok"; rm -rf "$X/$g"
  tail -1 "$LOGD/$g.log"
}
fail=0; pids=(); names=()
for g in "${GROUPS_ALL[@]}"; do run_group "$g" & pids+=($!); names+=("$g"); done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "失败 ${names[$i]}，见 $LOGD/${names[$i]}.log"; tail -3 "$LOGD/${names[$i]}.log"; fail=1; fi
done
[ "$fail" = 0 ] || { echo "MEMFRAMES=FAIL 有组失败"; exit 1; }

echo "阶段3：汇总校验"
"$N/.venv/bin/python" -I - "$OUT" "$D/meta/episode_manifest.json" <<'EOF'
import sys, json, pathlib, h5py
out, man = pathlib.Path(sys.argv[1]) / "episodes", json.load(open(sys.argv[2]))
bad, frames = [], 0
for e in man["episodes"]:
    p = out / f"ep_{e['global_episode_idx']:04d}.h5"
    try:
        with h5py.File(p, "r") as f:
            ok = f.attrs["complete"] and f.attrs["num_timesteps"] == e["num_timesteps"] and len(f["front_jpeg"]) == e["num_timesteps"]
            frames += int(f.attrs["num_timesteps"])
    except Exception as ex:
        ok = False
    if not ok: bad.append(e["global_episode_idx"])
tot = man["totals"]["timesteps"]
st = "PASS" if not bad and frames == tot else "FAIL"
print(f"MEMFRAMES={st} episodes={len(man['episodes'])-len(bad)}/1600 frames={frames} expect={tot} bad={bad[:5]}")
sys.exit(0 if st == "PASS" else 1)
EOF
