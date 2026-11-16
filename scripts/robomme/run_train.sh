#!/usr/bin/env bash
# NativeMEM RoboMME 4task Stage 2 训练 runner（1116 计划 C1、E 表 G2–G6a／G8／G8b／G12、J#9／#19）。
#
# 用法（由主会话放进 detached tmux，外层 tee 日志并写 EXIT_CODE=）：
#   bash scripts/robomme/run_train.sh --check-only     # CPU：G2 G3 G4 G5 G6a，全部 JAX_PLATFORMS=cpu
#   bash scripts/robomme/run_train.sh --smoke          # 8 卡：nm-smoke20，20 步，临时 save_interval 10／keep_period 20（G8）
#   bash scripts/robomme/run_train.sh --resume-smoke   # 8 卡：从 nm-smoke20 第 19 步 --resume 续训 2 步（G8b）
#   bash scripts/robomme/run_train.sh --formal         # 8 卡：nativemem-4task1600-b128-20k，正式超参（G12）
#
# 闸门（每个模式都跑，任一 FAIL 即退出非零）：
#   G2  TOKENIZER_PIN=PASS（verify_tokenizer_pin.py full 档，期望值来自 HF 服务端 pin）
#   G3  DATA_LINEAGE=EXCEPTION 固定行（用户开的 P-LINEAGE 例外，只对本 run）＋ DATA_SAME_AS_V17=PASS members=1600 mismatches=0
#   G4  NORM_STATS=PASS states=605611 action_vectors=30280550 ＋ NORM_SHA=PASS（文件已入 git 且与 HEAD 相同）
#   G5  WEIGHTS_LOAD=PASS …（CPU 单测）    G6a 合成数据单测（含跨 epoch 回绕）
#   smoke／resume-smoke／formal 另要求 G7 完成标记 MEMTOKENS_DONE.json；formal 另要求工作区干净且 HEAD 为 Beta 提交。
# 训练命令照计划：<fork venv>/bin/python scripts/train_nativemem.py nativemem_pi05_robomme4task --exp_name=<run>，
# runner 先 cd 到自身所在仓库根；wandb 关闭；XLA_PYTHON_CLIENT_MEM_FRACTION=0.9；缓存全部指向 /scratch/hongze/.cache。
set -euo pipefail

MODE="${1:-}"
case "$MODE" in --check-only|--smoke|--resume-smoke|--formal) ;; *)
  echo "用法: $0 --check-only|--smoke|--resume-smoke|--formal"; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/scripts/train_nativemem.py" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

CONFIG=nativemem_pi05_robomme4task
PY="${NM_PY:-/scratch/hongze/NativeMEM/.venv/bin/python}"
POLICY=/scratch/hongze/robomme_policy_learning_MotionJEPA
export NM_ROBOMME_DATA_ROOT="${NM_ROBOMME_DATA_ROOT:-/scratch/hongze/NativeMEM/data/robomme4task-v2}"
export NM_ROBOMME_EPISODE_MANIFEST="${NM_ROBOMME_EPISODE_MANIFEST:-$POLICY/v1-store/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json}"
export NM_STAGE1_PARAMS="${NM_STAGE1_PARAMS:-/scratch/hongze/NativeMEM/checkpoints/mem_tokenizer_pretrain/pretrain/49999/params}"
RECORDS="$POLICY/docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"
NORM_DIR="$REPO/assets/robomme4task"
CKPT_ROOT="$REPO/checkpoints/$CONFIG"
LOG_ROOT="$REPO/logs/train"
EXPECT_STATES=605611
EXPECT_ACTION_VECTORS=30280550
EXPECT_MEMBERS=1600

# 缓存逐项指向 /scratch/hongze/.cache（不覆盖 HOME）
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/scratch/hongze/.cache
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export HF_HOME=/scratch/hongze/.cache/huggingface
export OPENPI_DATA_HOME=/scratch/hongze/.cache/openpi
export JAX_COMPILATION_CACHE_DIR=/scratch/hongze/.cache/jax
export WANDB_DIR=/scratch/hongze/.cache/wandb WANDB_MODE=disabled
mkdir -p "$LOG_ROOT"

now() { TZ=America/New_York date '+%F %T 美东'; }
fail() { echo "$1"; echo "RUN_TRAIN=FAIL mode=$MODE reason=$2"; exit 1; }

echo "== run_train $MODE 开始 $(now) repo=$REPO head=$(git rev-parse HEAD)"
[ -x "$PY" ] || fail "错误: 解释器不存在 $PY" no_python

if [ "$MODE" = "--check-only" ]; then
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
fi
"$PY" -c "import openpi, jax; print('openpi.__file__=' + openpi.__file__); print('jax.devices()=' + str(jax.devices()))"
"$PY" -c "import openpi, pathlib, sys; sys.exit(0 if pathlib.Path(openpi.__file__).resolve().is_relative_to(pathlib.Path('$REPO').resolve()) else 1)" \
  || fail "错误: openpi 未指向本仓库 $REPO/src" openpi_path

# ---------------- G2 TOKENIZER_PIN ----------------
line=$(JAX_PLATFORMS=cpu "$PY" scripts/robomme/verify_tokenizer_pin.py --params-dir "$NM_STAGE1_PARAMS" --level full | tail -n 1 || true)
echo "$line"
case "$line" in TOKENIZER_PIN=PASS*) ;; *) fail "错误: G2 未通过" tokenizer_pin ;; esac

# ---------------- G3 DATA_LINEAGE（例外）＋ DATA_SAME_AS_V17 ----------------
echo "DATA_LINEAGE=EXCEPTION tokenizer=nativemem-stage1-49999 reason=user-choice-1116"
line=$(JAX_PLATFORMS=cpu "$PY" - "$NM_ROBOMME_DATA_ROOT/episodes" "$NM_ROBOMME_EPISODE_MANIFEST" "$RECORDS" "$EXPECT_MEMBERS" <<'PYEOF' | tail -n 1
# 每集 H5 的 attrs（member、h5_sha256、num_timesteps、exec_start_idx）逐条对 V17 训练库的 episode_map 与 episode_manifest。
import json, pathlib, sys
import h5py
ep_dir, manifest, records, expect = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), int(sys.argv[4])
eps = json.load(open(manifest))["episodes"]
by_key = {(e["h5_file"], e["raw_ep_idx"]): e for e in eps}
members = mismatches = 0
first = None
seen = set()
for task in ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"):
    for r in json.load(open(records / f"record_dataset_{task}_episode_map.json")):
        e = by_key[(f"record_dataset_{task}.h5", r["new_idx"])]
        g = e["global_episode_idx"]
        seen.add(g)
        members += 1
        p = ep_dir / f"ep_{g:04d}.h5"
        try:
            with h5py.File(p, "r") as f:
                a = {k: f.attrs[k] for k in ("member", "h5_sha256", "num_timesteps", "exec_start_idx", "global_episode_idx")}
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in a.items()}
            ok = (str(a["member"]) == r["member"] and str(a["h5_sha256"]) == r["h5_sha256"]
                  and int(a["num_timesteps"]) == int(r["timestep_count"]) == int(e["num_timesteps"])
                  and int(a["exec_start_idx"]) == int(e["exec_start_idx"]) and int(a["global_episode_idx"]) == g)
        except OSError as exc:
            ok, a = False, {"error": str(exc)}
        if not ok:
            mismatches += 1
            first = first or f"g={g} member={r['member']}"
if seen != set(range(len(eps))):
    mismatches += len(set(range(len(eps))) ^ seen)
ok = members == expect and mismatches == 0
print(f"DATA_SAME_AS_V17={'PASS' if ok else 'FAIL'} members={members} mismatches={mismatches}" + (f" first={first}" if first else ""))
PYEOF
) || true
echo "$line"
case "$line" in "DATA_SAME_AS_V17=PASS members=$EXPECT_MEMBERS mismatches=0"*) ;; *) fail "错误: G3 未通过" data_same_as_v17 ;; esac

# ---------------- G4 NORM_STATS／NORM_SHA ----------------
[ -f "$NORM_DIR/norm_stats.json" ] && [ -f "$NORM_DIR/norm_stats.provenance.json" ] || fail "错误: 缺 $NORM_DIR/norm_stats{,.provenance}.json" norm_missing
line=$("$PY" - "$NORM_DIR" "$EXPECT_STATES" "$EXPECT_ACTION_VECTORS" <<'PYEOF' | tail -n 1
import hashlib, json, pathlib, sys
d, es, ea = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sha = hashlib.sha256((d / "norm_stats.json").read_bytes()).hexdigest()
p = json.loads((d / "norm_stats.provenance.json").read_text())
ok = p["states"] == es and p["action_vectors"] == ea and p["norm_stats_sha256"] == sha
print(f"NORM_STATS={'PASS' if ok else 'FAIL'} states={p['states']} action_vectors={p['action_vectors']} sha256={sha}")
PYEOF
) || true
echo "$line"
case "$line" in "NORM_STATS=PASS states=$EXPECT_STATES action_vectors=$EXPECT_ACTION_VECTORS"*) ;; *) fail "错误: G4 未通过" norm_stats ;; esac
NORM_SHA=$(sha256sum "$NORM_DIR/norm_stats.json" | cut -d' ' -f1)
tracked=yes
for f in norm_stats.json norm_stats.provenance.json; do
  git ls-files --error-unmatch "assets/robomme4task/$f" >/dev/null 2>&1 || tracked=no
  git diff --quiet HEAD -- "assets/robomme4task/$f" 2>/dev/null || tracked=no
done
if [ "$tracked" = yes ] && [ -z "${NM_NORM_SHA256:-}" -o "${NM_NORM_SHA256:-}" = "$NORM_SHA" ]; then
  echo "NORM_SHA=PASS sha=${NORM_SHA:0:8} tracked=yes"
else
  echo "NORM_SHA=FAIL sha=${NORM_SHA:0:8} tracked=$tracked expect=${NM_NORM_SHA256:-HEAD}"; fail "错误: G4 NORM_SHA 未通过（须 git add -f 入库且与 HEAD 相同）" norm_sha
fi

# ---------------- G5 WEIGHTS_LOAD ＋ G6a 合成单测（CPU） ----------------
TEST_LOG="$LOG_ROOT/check-${MODE#--}-pytest.log"
set +e
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m pytest scripts/robomme/test_robomme_s1.py -q -s -p no:cacheprovider 2>&1 | tee "$TEST_LOG"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" = 0 ] || fail "错误: 单测失败（$TEST_LOG）" pytest
grep -q '^WEIGHTS_LOAD=PASS loader=NativeMEMWeightLoader src=stage1-49999 random_init=0 unused=encoder' "$TEST_LOG" \
  || fail "错误: G5 判定行缺失" weights_load
grep '^WEIGHTS_LOAD=PASS' "$TEST_LOG" | tail -n 1
echo "G6A_SYNTHETIC=PASS tests=$(grep -Eo '[0-9]+ passed' "$TEST_LOG" | tail -n 1)"

if [ "$MODE" = "--check-only" ]; then
  echo "RUN_TRAIN=PASS mode=check-only $(now)"
  exit 0
fi

# ---------------- GPU 模式的公共前置 ----------------
DONE_JSON="$NM_ROBOMME_DATA_ROOT/MEMTOKENS_DONE.json"
[ -f "$DONE_JSON" ] || fail "错误: 缺 G7 完成标记 $DONE_JSON" memtokens_not_done
grep -q 'MEMTOKENS=PASS' "$DONE_JSON" || fail "错误: $DONE_JSON 内无 MEMTOKENS=PASS" memtokens_verdict
echo "MEMTOKENS_DONE=PRESENT $(grep -o 'MEMTOKENS=PASS[^"]*' "$DONE_JSON")"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES

train() {  # $1 = run 名，其余为附加覆盖参数；训练日志单独 tee，返回训练进程退出码
  local run="$1"; shift
  local log="$LOG_ROOT/$run.train.log"
  echo "TRAIN_CMD: $PY scripts/train_nativemem.py $CONFIG --exp_name=$run --no-wandb-enabled $*"
  set +e
  "$PY" scripts/train_nativemem.py "$CONFIG" --exp_name="$run" --no-wandb-enabled "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "TRAIN_EXIT_CODE=$rc run=$run log=$log"
  return "$rc"
}

ckpt_steps() {  # 列出 run 目录下的数字 step 目录（按数值排序，空格分隔）；有 orbax tmp 目录时附 tmp 标记
  local d="$1"
  ( cd "$d" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -n | tr '\n' ' ' )
}

finite_check() {  # $1 = 训练日志，$2 = step 目录；输出 CHECK_FINITE 行
  "$PY" scripts/robomme/check_finite.py --log "$1" --ckpt "$2" --infer "$2" --config "$CONFIG" | tail -n 1 || true
}

case "$MODE" in
--smoke)
  RUN=nm-smoke20
  [ ! -e "$CKPT_ROOT/$RUN" ] || fail "错误: $CKPT_ROOT/$RUN 已存在（拒绝覆盖；由主会话按清单清理后重跑）" run_exists
  [ ! -e "$LOG_ROOT/$RUN.train.log" ] || fail "错误: $LOG_ROOT/$RUN.train.log 已存在（拒绝覆盖本轮日志）" log_exists
  # 冒烟临时把存档周期调进 20 步内：第 10 步存、第 19 步存后第 10 步被轮转删除（keep_period 20、max_to_keep 1），正式超参不变。
  set +e; train "$RUN" --num_train_steps=20 --save_interval=10 --keep_period=20 --log_interval=1; rc=$?; set -e
  steps=$(ckpt_steps "$CKPT_ROOT/$RUN")
  saves=$(grep -c 'Waiting for checkpoint manager to finish' "$LOG_ROOT/$RUN.train.log" || true)
  rotate=FAIL; [ "$steps" = "19 " ] && [ "$saves" = 2 ] && rotate=PASS
  nsteps=$(grep -Eo 'Step [0-9]+:' "$LOG_ROOT/$RUN.train.log" | sort -u | wc -l)
  fin=FAIL; save_load=FAIL
  if [ "$rc" = 0 ] && [ -d "$CKPT_ROOT/$RUN/19" ]; then
    fl=$(finite_check "$LOG_ROOT/$RUN.train.log" "$CKPT_ROOT/$RUN/19"); echo "$fl"
    case "$fl" in *finite=PASS*) fin=PASS ;; esac
    case "$fl" in *ckpt=PASS*) save_load=PASS ;; esac
  fi
  verdict=FAIL
  [ "$rc" = 0 ] && [ "$nsteps" = 20 ] && [ "$rotate" = PASS ] && [ "$fin" = PASS ] && [ "$save_load" = PASS ] && verdict=PASS
  echo "SMOKE20=$verdict steps=$nsteps save_load=$save_load rotate=$rotate exit=$rc finite=$fin ckpt_dirs=[$steps] saves=$saves"
  [ "$verdict" = PASS ] || exit 1
  ;;
--resume-smoke)
  RUN=nm-smoke20
  [ -d "$CKPT_ROOT/$RUN/19" ] || fail "错误: 缺 $CKPT_ROOT/$RUN/19（先跑 --smoke）" no_smoke_ckpt
  # 续训 2 步（第 20、21 步），keep_period=1 让第 19 步存档保留下来供核对 Adam／EMA 恢复；数据流不恢复属原版行为。
  set +e; train "$RUN" --resume --num_train_steps=22 --save_interval=10 --keep_period=1 --log_interval=1; rc=$?; set -e
  first_step=$(grep -Eo 'Step [0-9]+:' "$LOG_ROOT/$RUN.train.log" | tail -n 2 | head -n 1 | grep -Eo '[0-9]+' || true)
  rl=FAIL
  if [ "$rc" = 0 ] && [ -d "$CKPT_ROOT/$RUN/21" ]; then
    rl=$("$PY" scripts/robomme/check_finite.py --resume-check "$CKPT_ROOT/$RUN" --from-step 19 --to-step 21 | tail -n 1 || true)
    echo "$rl"
  fi
  case "$rl" in RESUME=PASS*) [ "$first_step" = 20 ] && { echo "RESUME=PASS first_step=$first_step exit=$rc"; exit 0; } ;; esac
  echo "RESUME=FAIL first_step=${first_step:-none} exit=$rc"; exit 1
  ;;
--formal)
  RUN=nativemem-4task1600-b128-20k
  [ -z "$(git status --porcelain --ignore-submodules=dirty)" ] || { git status --short; fail "错误: 工作区不干净，正式训练必须从 clean HEAD 起跑" dirty; }
  subj=$(git log -1 --format=%s)
  case "$subj" in *Beta*) ;; *) fail "错误: HEAD 不是 Beta 提交（$subj）" not_beta ;; esac
  echo "BETA_HEAD=$(git rev-parse HEAD) subject=$subj"
  [ ! -e "$CKPT_ROOT/$RUN" ] || fail "错误: $CKPT_ROOT/$RUN 已存在（拒绝覆盖）" run_exists
  [ ! -e "$LOG_ROOT/$RUN.train.log" ] || fail "错误: $LOG_ROOT/$RUN.train.log 已存在（拒绝覆盖本轮日志）" log_exists
  set +e; train "$RUN"; rc=$?; set -e
  steps=$(ckpt_steps "$CKPT_ROOT/$RUN")
  fin=FAIL
  if [ "$rc" = 0 ] && [ "$steps" = "19999 " ]; then
    fl=$(finite_check "$LOG_ROOT/$RUN.train.log" "$CKPT_ROOT/$RUN/19999"); echo "$fl"
    case "$fl" in *finite=PASS*) fin=PASS ;; esac
  fi
  verdict=FAIL; [ "$rc" = 0 ] && [ "$steps" = "19999 " ] && [ "$fin" = PASS ] && verdict=PASS
  out="TRAIN_FINAL=$verdict step=19999 ckpts=$(echo $steps | wc -w) finite=$fin exit=$rc ckpt_dirs=[$steps]"
  echo "$out"
  echo "$out" > "$LOG_ROOT/$RUN.TRAIN_FINAL.verdict"
  [ "$verdict" = PASS ] || exit 1
  ;;
esac
echo "RUN_TRAIN=PASS mode=${MODE#--} $(now)"
