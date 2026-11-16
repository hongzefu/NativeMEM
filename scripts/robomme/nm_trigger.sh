#!/usr/bin/env bash
# NativeMEM 第 4 步自动起跑监听（1116 计划 §6、J#22；由主会话放进 detached tmux nm-trig 运行）。
# 四个条件同时成立才起跑，且只起一次：
#   ① $NM_ROOT/runs/READY.json 存在（本轮 CPU 准备全部通过，由主会话写）；
#   ② tmux 会话 '=v17-eval' 不存在，且 V17 评估日志最后一个非空行是完整的 EXIT_CODE=<数字>，
#      日志内能找到 RUN_NAME=<V17_RUN_NAME> 或 run=<V17_RUN_NAME>（只认身份，不看成败）；
#   ③ nvidia-smi 列出 NGPU 张卡且每张 memory.used < MEM_MAX_MIB；
#   ④ mkdir $NM_ROOT/runs/nm-trigger.lock 原子创建成功（防重复起跑）。
# 每轮间隔 POLL_S 秒（nvidia-smi 也只在每轮查一次，默认 60 s，只读查询）。状态与上一轮不同才写一行日志，供 Monitor 过滤。
# 成立后执行 NM_TRIG_CMD（默认只 echo，不实际起跑；由主会话决定挂什么命令），记录其退出码后退出。
# 用法：nm_trigger.sh [--once]      --once：只评估一轮并打印状态，不抢锁、不执行命令（主会话自检用）
# 可覆盖：NM_ROOT、POLICY_SRC、V17_SESSION（默认 v17-eval）、V17_LOG（默认 $POLICY_SRC/v1-store/logs/v17-eval.log）、
#        V17_RUN_NAME（默认 local692-ctx-motion160-50k）、NGPU（8）、MEM_MAX_MIB（2000）、POLL_S（60）、
#        NM_TRIG_LOG（默认 $NM_ROOT/runs/logs/nm-trigger.log）、NM_TRIG_CMD、NVIDIA_SMI（默认 nvidia-smi）
# 日志关键行：NM_TRIG_STATE …、NM_TRIGGER=FIRED、NM_TRIGGER=REFUSED reason=lock_exists、NM_TRIGGER_CMD_EXIT=<rc>
set -uo pipefail
NM_ROOT="${NM_ROOT:-/scratch/hongze/NativeMEM}"
POLICY_SRC="${POLICY_SRC:-/scratch/hongze/robomme_policy_learning_MotionJEPA}"
READY="$NM_ROOT/runs/READY.json"
LOCK="$NM_ROOT/runs/nm-trigger.lock"
V17_SESSION="${V17_SESSION:-v17-eval}"
V17_LOG="${V17_LOG:-$POLICY_SRC/v1-store/logs/v17-eval.log}"
V17_RUN_NAME="${V17_RUN_NAME:-local692-ctx-motion160-50k}"
NGPU="${NGPU:-8}"
MEM_MAX_MIB="${MEM_MAX_MIB:-2000}"
POLL_S="${POLL_S:-60}"
LOG="${NM_TRIG_LOG:-$NM_ROOT/runs/logs/nm-trigger.log}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
NM_TRIG_CMD="${NM_TRIG_CMD:-echo 'NM_TRIGGER_DRY 默认命令只打印、不起跑；主会话用 NM_TRIG_CMD 指定第 4 步命令'}"
ONCE=0; [[ "${1:-}" == --once ]] && ONCE=1

policy_real="$(realpath -m "$POLICY_SRC")"
for p in "$LOG" "$LOCK" "$READY"; do
    real="$(realpath -m "$p")"
    [[ "$real" != "$policy_real" && "$real" != "$policy_real"/* ]] || { echo "可写路径落在 policy 仓，拒绝：$p"; exit 2; }
done
[[ "$POLL_S" =~ ^[1-9][0-9]*$ && "$NGPU" =~ ^[1-9][0-9]*$ && "$MEM_MAX_MIB" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_S/NGPU/MEM_MAX_MIB 必须是正整数"; exit 2; }
mkdir -p "$(dirname "$LOG")"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# ── 四个条件（前三个只读）──
cond_ready() { [[ -f "$READY" ]]; }
cond_v17() {  # 打印细节；成立返回 0
    if tmux has-session -t "=$V17_SESSION" 2>/dev/null; then echo "session_alive"; return 1; fi
    [[ -f "$V17_LOG" ]] || { echo "log_missing"; return 1; }
    local last
    last="$(grep -v '^[[:space:]]*$' "$V17_LOG" | tail -n 1)"
    [[ "$last" =~ ^EXIT_CODE=[0-9]+$ ]] || { echo "no_exit_code_tail"; return 1; }
    grep -Eq "(RUN_NAME=|run=)${V17_RUN_NAME}([^A-Za-z0-9_-]|\$)" "$V17_LOG" || { echo "run_name_mismatch"; return 1; }
    echo "done(${last})"
}
cond_gpu() {  # 打印细节；成立返回 0
    local csv n bad
    csv="$("$NVIDIA_SMI" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || { echo "query_failed"; return 1; }
    n="$(grep -c . <<< "$csv")"
    bad="$(awk -F', *' -v m="$MEM_MAX_MIB" 'NF>=2 && ($2+0) >= m {printf "%s:%s ", $1, $2}' <<< "$csv")"
    [[ "$n" == "$NGPU" ]] || { echo "gpus=$n!=$NGPU"; return 1; }
    [[ -z "$bad" ]] || { echo "busy[${bad% }]"; return 1; }
    echo "idle(max<$MEM_MAX_MIB)"
}

log "NM_TRIG_START once=$ONCE ready=$READY v17_session=$V17_SESSION v17_log=$V17_LOG v17_run=$V17_RUN_NAME ngpu=$NGPU mem_max_mib=$MEM_MAX_MIB poll_s=$POLL_S lock=$LOCK cmd=$NM_TRIG_CMD"
PREV=""
while true; do
    R=0; cond_ready && R=1
    V="$(cond_v17)"; VOK=$?
    G="$(cond_gpu)"; GOK=$?
    STATE="ready=$R v17=$V gpu=$G"
    if [[ "$STATE" != "$PREV" || "$ONCE" == 1 ]]; then
        log "NM_TRIG_STATE $STATE"
        PREV="$STATE"
    fi
    if [[ "$R" == 1 && "$VOK" == 0 && "$GOK" == 0 ]]; then
        if [[ "$ONCE" == 1 ]]; then log "NM_TRIG_ONCE all_conditions=1（--once 不抢锁、不执行命令）"; exit 0; fi
        if ! mkdir "$LOCK" 2>/dev/null; then
            log "NM_TRIGGER=REFUSED reason=lock_exists lock=$LOCK"
            exit 3
        fi
        printf 'pid=%s host=%s utc=%s state=%s\n' "$$" "$(hostname)" "$(date -u +%FT%TZ)" "$STATE" > "$LOCK/owner.txt"
        log "NM_TRIGGER=FIRED $STATE"
        bash -c "$NM_TRIG_CMD" 2>&1 | tee -a "$LOG"
        RC=${PIPESTATUS[0]}
        log "NM_TRIGGER_CMD_EXIT=$RC"
        exit "$RC"
    fi
    if [[ "$ONCE" == 1 ]]; then log "NM_TRIG_ONCE all_conditions=0"; exit 1; fi
    sleep "$POLL_S"
done
