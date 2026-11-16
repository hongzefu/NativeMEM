#!/usr/bin/env bash
# NativeMEM GPU 阶段串联驱动（1116 计划第 4–7 步；由 nm_trigger.sh 在四条件成立后执行：
#   NM_TRIG_CMD="bash scripts/robomme/nm_pipeline.sh"）。
#
# 步骤（按序；任一步 FAIL 立即非 0 退出并打印 NM_PIPELINE=FAIL step=<名> reason=…，不自动重试）：
#   第 4 步  memtok         extract_mem_tokens.py --init → 8 片（第 i 片 CUDA_VISIBLE_DEVICES=i）→ --finalize（G7）
#   第 5 步  dataset_smoke  run_train.sh --dataset-smoke          DATASET_SMOKE=PASS n=605611（G6b）
#            smoke20        run_train.sh --smoke                  SMOKE20=PASS … rotate=PASS exit=0 finite=PASS（G8）
#            resume         run_train.sh --resume-smoke           RESUME=PASS（G8b）
#            online_token   check_online_token.py run（单卡）     ENCODER_IMPL=PASS（硬闸门，G9a）＋ ONLINE_TOKEN=RECORDED（G9b）
#            g6b            local_692_pool_nativemem.sh --g6b <nm-smoke20/19> 7       EVAL_LOAD_SMOKE=PASS（G10）
#            capacity       --g6b-capacity <nm-smoke20/19>：先每卡 4 份；只有日志出现 EVAL_CAPACITY=FAIL per_gpu=4 才退 3 份，
#                           其他非 0 一律 die capacity_rc；选定值写 runs/pipeline_state.json（G11）
#            cleanup_smoke  删冒烟存档目录 checkpoints/nativemem_pi05_robomme4task/nm-smoke20（resume 同目录），前后 ls -ld
#   第 6 步  formal         HEAD subject 含 Beta 且工作区干净 → run_train.sh --formal   TRAIN_FINAL=PASS step=19999 ckpts=1 finite=PASS（G12）
#                           通过后 detached tmux nm-ckptup 起 upload_ckpt.sh（日志 logs/nm-ckptup.log，不阻塞）
#   第 7 步  eval           local_692_pool_nativemem.sh <19999 存档> 7（EVAL_PER_GPU 取 pipeline_state）  EVAL692=PASS（G14）
# 每步开始／结束打印 NM_STEP=<名> start|done <美东时间>；判定行原样打到 stdout，同时写 logs/pipeline/<步名>.log。
# 每步开始前只读查询 8 卡显存并打印；只有 memtok 要求 8 卡均 < 2000 MiB。
#
# 可覆盖：NM_PIPELINE_FROM=<步名>（从该步续跑，之前的步骤跳过）、DRY_RUN=1（只打印将执行的命令与期望判定行）、
#        NM_PIPELINE_LOGDIR（默认 <仓库>/logs/pipeline）、NM_PIPELINE_STATE（默认 <仓库>/runs/pipeline_state.json）、
#        NM_PIPELINE_STUB=<可执行文件>（单测用：所有外部命令改为执行「<STUB> <步名> <原命令…>」）。
# 冒烟、压测、评估各脚本自带「拒绝覆盖」：续跑前由主会话按清单清理对应产物。
set -Eeuo pipefail
# 任何未被 || die 接住的非 0 退出（set -e 触发）都留下一行带步名的 FAIL 判定行（审查 F4）
trap 'echo "NM_PIPELINE=FAIL step=${CUR:-init} reason=unexpected_rc rc=$? line=$LINENO"' ERR

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${NM_PY:-/scratch/hongze/NativeMEM/.venv/bin/python}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/scratch/hongze/.cache/uv JAX_COMPILATION_CACHE_DIR=/scratch/hongze/.cache/jax
CONFIG=nativemem_pi05_robomme4task
CKPT_ROOT="$REPO/checkpoints/$CONFIG"
SMOKE_DIR="$CKPT_ROOT/nm-smoke20"
FORMAL_RUN=nativemem-4task1600-b128-20k
FORMAL_CKPT="$CKPT_ROOT/$FORMAL_RUN/19999"
DATA_ROOT="${NM_ROBOMME_DATA_ROOT:-/scratch/hongze/NativeMEM/data/robomme4task-v2}"
LOGDIR="${NM_PIPELINE_LOGDIR:-$REPO/logs/pipeline}"
STATE="${NM_PIPELINE_STATE:-$REPO/runs/pipeline_state.json}"
DRY="${DRY_RUN:-0}"
STUB="${NM_PIPELINE_STUB:-}"
FROM="${NM_PIPELINE_FROM:-}"
SEED=7
STEPS=(memtok dataset_smoke smoke20 resume online_token g6b capacity cleanup_smoke formal eval)
mkdir -p "$LOGDIR" "$(dirname "$STATE")"

now() { TZ=America/New_York date '+%F %T 美东'; }
CUR=""
die() { echo "NM_PIPELINE=FAIL step=${CUR:-init} reason=$1"; exit 1; }

# 外部命令统一入口：DRY 只打印；STUB 时交给打桩脚本
x() {
    if [[ "$DRY" == 1 ]]; then echo "DRY_CMD[$CUR]: $*"; return 0; fi
    if [[ -n "$STUB" ]]; then "$STUB" "$CUR" "$@"; return $?; fi
    "$@"
}
# 跑一个子命令：输出 tee 进 <步名>.log，返回命令自身退出码
xlog() {
    # 整组放在 || 左侧：命令失败不触发 errexit／ERR trap，由调用方按返回码决定（|| die 或回退）
    local rc=0
    { x "$@" 2>&1 | tee -a "$LOGDIR/$CUR.log"; rc=${PIPESTATUS[0]}; } || true
    return "$rc"
}
# 判定行：从本步日志里取匹配 $1（ERE，行首锚定）的最后一行，原样打印；无则 FAIL
expect() {
    local pat="$1" reason="$2" line
    if [[ "$DRY" == 1 ]]; then echo "DRY_EXPECT[$CUR]: ^$pat"; return 0; fi
    line="$(grep -E "^$pat" "$LOGDIR/$CUR.log" | tail -n 1 || true)"
    [[ -n "$line" ]] || die "$reason"
    echo "VERDICT[$CUR] $line"
}
gpu_report() {  # 只读；$1=1 时要求 8 卡 < 2000 MiB
    local csv bad
    if [[ "$DRY" == 1 || -n "$STUB" ]]; then echo "GPU_CHECK[$CUR] skipped(dry/stub) require_idle=$1"; return 0; fi
    csv="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)" || die gpu_query_failed
    echo "GPU_MEM[$CUR] $(tr '\n' ' ' <<< "$csv")"
    if [[ "$1" == 1 ]]; then
        bad="$(awk -F', *' '($2+0) >= 2000 {printf "%s:%s ", $1, $2}' <<< "$csv")"
        [[ "$(grep -c . <<< "$csv")" == 8 && -z "$bad" ]] || die "gpu_not_idle[${bad% }]"
    fi
}
smoke_ckpt() {  # 固定用冒烟第 19 步存档（resume 之后目录里还有 21，那是续训产物，不用于评估冒烟；审查 F1）
    if [[ "$DRY" == 1 || -n "$STUB" ]]; then echo "$SMOKE_DIR/19"; return 0; fi
    [[ -d "$SMOKE_DIR/19" ]] || return 1
    echo "$SMOKE_DIR/19"
}

# ── 各步 ──
step_memtok() {
    local i pid rc pids=() shard_of=() failed=""
    xlog env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" scripts/robomme/extract_mem_tokens.py --init --num-shards 8 \
        --data-root "$DATA_ROOT" || die memtok_init
    for i in 0 1 2 3 4 5 6 7; do
        if [[ "$DRY" == 1 ]]; then
            x env CUDA_VISIBLE_DEVICES="$i" "$PY" scripts/robomme/extract_mem_tokens.py --num-shards 8 --shard-index "$i" \
                --data-root "$DATA_ROOT"
            continue
        fi
        # 不套子 shell：$! 就是实际进程（env 会 exec 成 python；打桩时是打桩脚本），失败时才能按确切 PID 收掉
        ${STUB:+"$STUB" "$CUR"} env CUDA_VISIBLE_DEVICES="$i" "$PY" scripts/robomme/extract_mem_tokens.py --num-shards 8 \
            --shard-index "$i" --data-root "$DATA_ROOT" > "$LOGDIR/memtok-shard$i.log" 2>&1 &
        pid=$!
        pids+=("$pid"); shard_of[$pid]=$i
        echo "MEMTOK_SHARD_STARTED shard=$i pid=$pid log=$LOGDIR/memtok-shard$i.log"
    done
    # 谁先结束先收谁；任一片失败立即按本脚本记录的确切 PID 结束其余分片（只杀自己起的进程；审查 F3）
    local -A running=()
    for pid in "${pids[@]}"; do running[$pid]=1; done
    while (( ${#running[@]} > 0 )); do
        rc=0; wait -n -p pid "${!running[@]}" || rc=$?
        unset "running[$pid]"
        echo "MEMTOK_SHARD_EXIT shard=${shard_of[$pid]} pid=$pid rc=$rc"
        if (( rc != 0 )); then
            failed="${shard_of[$pid]}"
            for pid in "${!running[@]}"; do
                echo "MEMTOK_SHARD_KILL shard=${shard_of[$pid]} pid=$pid"
                kill "$pid" 2>/dev/null || true
            done
            for pid in "${!running[@]}"; do wait "$pid" 2>/dev/null || true; done
            die "memtok_shard_failed[$failed]"
        fi
    done
    xlog env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/robomme/extract_mem_tokens.py --finalize --num-shards 8 \
        --data-root "$DATA_ROOT" || die memtok_finalize
    expect 'MEMTOKENS=PASS rows=1192918 views=2 dim=1152 nonfinite=0 zero_rows=0 shards=8/8 gaps=0 overlaps=0 recompute=' memtokens
}
step_dataset_smoke() {
    xlog bash scripts/robomme/run_train.sh --dataset-smoke || die dataset_smoke_rc
    expect 'DATASET_SMOKE=PASS n=605611( |$)' dataset_smoke
}
step_smoke20() {
    xlog bash scripts/robomme/run_train.sh --smoke || die smoke20_rc
    expect 'SMOKE20=PASS .*rotate=PASS exit=0 finite=PASS' smoke20
}
step_resume() {
    xlog bash scripts/robomme/run_train.sh --resume-smoke || die resume_rc
    expect 'RESUME=PASS' resume
}
step_online_token() {
    xlog env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/robomme/check_online_token.py run --data-root "$DATA_ROOT" \
        --memtokens "$DATA_ROOT/mem_tokens.f32.bin" --report "$LOGDIR/online_token.json" || die online_token_rc
    expect 'ENCODER_IMPL=PASS' encoder_impl
    expect 'ONLINE_TOKEN=RECORDED' online_token_recorded
}
step_g6b() {
    local ck
    ck="$(smoke_ckpt)" || die "no_smoke_ckpt19($SMOKE_DIR/19)"
    xlog bash scripts/robomme/local_692_pool_nativemem.sh --g6b "$ck" "$SEED" || die g6b_rc
    expect 'EVAL_LOAD_SMOKE=PASS' eval_load_smoke
}
step_capacity() {
    local ck per=""
    ck="$(smoke_ckpt)" || die "no_smoke_ckpt19($SMOKE_DIR/19)"
    local rc4=0 rc3=0
    xlog env EVAL_PER_GPU=4 bash scripts/robomme/local_692_pool_nativemem.sh --g6b-capacity "$ck" "$SEED" || rc4=$?
    if [[ "$DRY" == 1 ]]; then
        per=4
    elif (( rc4 == 0 )) && grep -Eq '^EVAL_CAPACITY=PASS per_gpu=4( |$)' "$LOGDIR/$CUR.log"; then
        per=4
    elif grep -Eq '^EVAL_CAPACITY=FAIL per_gpu=4( |$)' "$LOGDIR/$CUR.log"; then
        # 只有明确的容量不足才退到每卡 3 份（审查 F2）
        echo "CAPACITY_FALLBACK per_gpu=4 容量不足（EVAL_CAPACITY=FAIL per_gpu=4），改压每卡 3 份"
        xlog env EVAL_PER_GPU=3 bash scripts/robomme/local_692_pool_nativemem.sh --g6b-capacity "$ck" "$SEED" || rc3=$?
        if (( rc3 == 0 )) && grep -Eq '^EVAL_CAPACITY=PASS per_gpu=3( |$)' "$LOGDIR/$CUR.log"; then
            per=3
        elif grep -Eq '^EVAL_CAPACITY=FAIL per_gpu=3( |$)' "$LOGDIR/$CUR.log"; then
            die capacity_4_and_3_failed
        else
            die "capacity_rc(per_gpu=3 rc=$rc3)"
        fi
    else
        die "capacity_rc(per_gpu=4 rc=$rc4)"
    fi
    expect "EVAL_CAPACITY=PASS per_gpu=$per" eval_capacity
    if [[ "$DRY" == 1 ]]; then echo "DRY_WRITE[$CUR]: $STATE per_gpu=$per"; return 0; fi
    printf '{"per_gpu": %s, "shards": %s, "written_utc": "%s"}\n' "$per" "$((8 * per))" "$(date -u +%FT%TZ)" > "$STATE"
    echo "PIPELINE_STATE per_gpu=$per shards=$((8 * per)) file=$STATE"
}
step_cleanup_smoke() {
    local d="${SMOKE_DIR:?}"
    [[ "$d" == "$REPO/checkpoints/$CONFIG/nm-smoke20" ]] || die "smoke_dir_unexpected($d)"
    echo "LS_BEFORE: $(ls -ld "$d" 2>&1 || true)"
    x rm -rf -- "$d" || die smoke_rm
    echo "LS_AFTER: $(ls -ld "$d" 2>&1 || true)"
    if [[ "$DRY" != 1 && -z "$STUB" && -e "$d" ]]; then die smoke_dir_still_exists; fi
    echo "SMOKE_CLEANUP=PASS dir=$d"
}
step_formal() {
    local subj dirty
    subj="$(git log -1 --format=%s)" || die git_log_failed
    dirty="$(git status --porcelain)" || die git_status_failed
    echo "HEAD_CHECK sha=$(git rev-parse --short HEAD) subject=$subj dirty_lines=$(grep -c . <<< "$dirty" || true)"
    if [[ "$DRY" != 1 ]]; then
        [[ "$subj" == *Beta* && -z "$dirty" ]] || die need_beta
    fi
    xlog bash scripts/robomme/run_train.sh --formal || die formal_rc
    expect 'TRAIN_FINAL=PASS step=19999 ckpts=1 finite=PASS' train_final
    if [[ "$DRY" != 1 && -z "$STUB" ]] && tmux has-session -t '=nm-ckptup' 2>/dev/null; then die ckptup_session_exists; fi
    x tmux new-session -d -s nm-ckptup \
        "cd '$REPO'; set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/robomme/upload_ckpt.sh 2>&1 | tee '$REPO/logs/nm-ckptup.log'; echo \"EXIT_CODE=\$?\" >> '$REPO/logs/nm-ckptup.log'" \
        || die ckptup_launch
    echo "NM_CKPTUP=STARTED tmux=nm-ckptup log=$REPO/logs/nm-ckptup.log（不阻塞，完成行 HF_VERIFY=PASS ckpts=1）"
}
step_eval() {
    local per
    if [[ "$DRY" == 1 ]]; then per="<pipeline_state.per_gpu>"; else
        [[ -f "$STATE" ]] || die no_pipeline_state
        per="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['per_gpu'])" "$STATE")" || die bad_pipeline_state
        [[ "$per" == 4 || "$per" == 3 ]] || die "bad_per_gpu($per)"
    fi
    echo "EVAL_PLAN per_gpu=$per ckpt=$FORMAL_CKPT"
    xlog env EVAL_PER_GPU="$per" bash scripts/robomme/local_692_pool_nativemem.sh "$FORMAL_CKPT" "$SEED" || die eval_rc
    expect 'EVAL692=PASS' eval692
}

# ── 主流程 ──
if [[ -n "$FROM" ]]; then
    [[ " ${STEPS[*]} " == *" $FROM "* ]] || { echo "NM_PIPELINE=FAIL step=init reason=unknown_from($FROM) steps=${STEPS[*]}"; exit 2; }
fi
echo "NM_PIPELINE_START $(now) repo=$REPO head=$(git rev-parse --short HEAD) from=${FROM:-memtok} dry=$DRY stub=${STUB:-none} logdir=$LOGDIR"
started=0; [[ -z "$FROM" ]] && started=1
for s in "${STEPS[@]}"; do
    [[ "$started" == 1 || "$s" == "$FROM" ]] || { echo "NM_STEP=$s skipped(from=$FROM)"; continue; }
    started=1
    CUR="$s"
    if [[ -e "$LOGDIR/$s.log" && "$DRY" != 1 ]]; then mv "$LOGDIR/$s.log" "$LOGDIR/$s.log.prev-$(date -u +%Y%m%dT%H%M%SZ)"; fi
    echo "NM_STEP=$s start $(now)"
    req=0; [[ "$s" == memtok ]] && req=1
    gpu_report "$req"
    "step_$s"
    echo "NM_STEP=$s done $(now)"
done
echo "NM_PIPELINE=PASS $(now)"
