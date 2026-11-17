#!/usr/bin/env bash
# 跑一个分片：起一次 NativeMEM 服务器（fork .venv 的 serve_nativemem.py），分块多次起客户端副本，收尾验收。
# 由 policy 仓 scripts/evaluation/run_shard.sh 复制改写（1116 计划 C2 第 3 行、J#6 双根路径）：
#   - 不 source policy 仓 env.sh，改 source 本目录 eval_env.sh；一切输出落 $NM_EVAL_ROOT/<运行名>/<分片>；
#   - 服务器换成 serve_nativemem.py，客户端换成 scripts/robomme/client/eval.py（--args.obs_horizon=48，hunk ④）；
#   - 去掉 check_checkpoint.py 与 motion 变量，改为检查存档含 params/ 与 assets/robomme4task/norm_stats.json；
#   - policy 仓脚本（verify_sources.py、render_probe.py、check_shard.py）一律绝对路径只读调用。
# 用法：run_shard_nativemem.sh <运行名> <checkpoint绝对路径> <分片计划json> [策略seed=7] [分片标识]
# 可覆盖：EVAL_TIMEOUT（默认 7200）、CHUNK_EPISODES（默认 20）、ALLOW_RESUME=1、
#        POLICY（结果目录层级名，默认 nativemem-4task1600-b128-20k）、POLICY_CONFIG（默认 nativemem_pi05_robomme4task）、
#        EXPECT_CKPT_ID（默认 19999）、EPISODE_WALL_S、SHARD_INDEX、DEMO_PREFIX_STORE、EXPECT_DEMO_INJECT=1、
#        LOCAL_GPU_SHARE / LOCAL_GPU_SLOT_FILE（同原脚本）、DRY_RUN=1（只打印解析后的路径并断言，不起任何进程）
set -euo pipefail
source "$(dirname "$0")/eval_env.sh"
RUN_NAME="${1:?需要运行名}"
CKPT="${2:?需要 checkpoint 绝对路径}"
PLAN="${3:?需要分片计划 json}"
SEED="${4:-7}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_TAG="${5:-s$SHARD_INDEX}"
POLICY="${POLICY:-nativemem-4task1600-b128-20k}"
POLICY_CONFIG="${POLICY_CONFIG:-nativemem_pi05_robomme4task}"
EXPECT_CKPT_ID="${EXPECT_CKPT_ID:-19999}"
EPISODE_WALL_S="${EPISODE_WALL_S:-}"
MAX_STEPS=2000
OBS_HORIZON=48
EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"
CHUNK_EPISODES="${CHUNK_EPISODES:-20}"
DRY="${DRY_RUN:-0}"
_gpu_owner_ok() {  # $1=GPU 上的 PID，$2=已登记根（空格分隔）；沿 PPid 链能走到某个根即返回 0
    local q="$1" roots="$2"
    while [[ -n "$q" && "$q" != 0 && "$q" != 1 && " $roots " != *" $q "* ]]; do
        q="$(awk '/^PPid:/ {print $2}' "/proc/$q/status" 2>/dev/null || true)"
    done
    [[ -n "$q" && " $roots " == *" $q "* ]]
}
check_local_gpu() {
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo '本机必须显式选择 GPU'; return 2; }
    local gpu_pids
    gpu_pids="$("${NVIDIA_SMI:-nvidia-smi}" -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" || return 2
    if [[ "${LOCAL_GPU_SHARE:-1}" == 1 ]]; then
        if [[ -n "$gpu_pids" ]]; then
            echo "所选 GPU 已被占用，停止起跑：$gpu_pids"
            return 2
        fi
        echo "LOCAL_GPU_IDLE=PASS device=$CUDA_VISIBLE_DEVICES"
        return 0
    fi
    # LOCAL_GPU_SHARE=N>=2：只认 local_692_pool_nativemem.sh 在 flock 下写的每卡占位文件（逻辑同原脚本）
    [[ "${LOCAL_GPU_SHARE}" =~ ^[0-9]+$ && "${LOCAL_GPU_SHARE}" -ge 2 ]] || { echo "LOCAL_GPU_SHARE 只能是 1 或 >=2 的整数：$LOCAL_GPU_SHARE"; return 2; }
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo "LOCAL_GPU_SHARE=$LOCAL_GPU_SHARE 只支持单卡：$CUDA_VISIBLE_DEVICES"; return 2; }
    local slot_file="${LOCAL_GPU_SLOT_FILE:-}" slots roots p q n_reg own=0 tag pid
    [[ -n "$slot_file" && -f "$slot_file" ]] || { echo "LOCAL_GPU_SHARE=$LOCAL_GPU_SHARE 需要占位文件 LOCAL_GPU_SLOT_FILE：${slot_file:-unset}"; return 2; }
    slots="$(flock -s "$slot_file.lock" cat "$slot_file")" || return 2
    n_reg=0; roots="$$"
    while read -r tag pid; do
        [[ -n "$tag" ]] || continue
        n_reg=$((n_reg + 1))
        if [[ "$tag" == "$SHARD_TAG" ]]; then own=1; fi
        if [[ "$pid" =~ ^[0-9]+$ ]]; then roots="$roots $pid"; fi
    done <<< "$slots"
    [[ "$own" == 1 ]] || { echo "本片 $SHARD_TAG 未在占位文件登记，停止起跑：$slot_file"; return 2; }
    (( n_reg <= LOCAL_GPU_SHARE )) || { echo "本卡已登记 $n_reg 片 > $LOCAL_GPU_SHARE，停止起跑：$slot_file"; return 2; }
    # 沿 PPid 链向上找已登记根。修复（g6b 实测误判）：nvidia-smi 列出的进程可能在查询之后、读 /proc 之前退出
    # （同卡另一片的 render_probe／客户端、或驱动尚未回收的刚退出进程），原逻辑读到空 PPid 就判「不属于已登记树」。
    # 现在：未找到根时，若该 PID 已不存在 → 它不再占卡，记录后跳过；仍存活 → 重走一遍（防中途祖先退出的瞬态），
    # 仍不属于已登记树才拒绝。活着的陌生进程照旧拒绝起跑。
    for p in $gpu_pids; do
        [[ "$p" =~ ^[0-9]+$ ]] || { echo "nvidia-smi 返回非数字 PID：$p，停止起跑"; return 2; }
        _gpu_owner_ok "$p" "$roots" && continue
        if [[ ! -e "/proc/$p" ]]; then
            echo "GPU_PROC_EXITED pid=$p（查询后已退出，不计入占用）"
            continue
        fi
        _gpu_owner_ok "$p" "$roots" && continue
        echo "所选 GPU 上的计算进程 $p 不属于已登记分片的进程树（根=$roots），停止起跑"
        return 2
    done
    echo "LOCAL_GPU_SHARE=PASS device=$CUDA_VISIBLE_DEVICES share=$LOCAL_GPU_SHARE registered=$n_reg gpu_procs=$(wc -w <<< "$gpu_pids")"
}
# 单测入口：只跑 GPU 归属检查后退出（NVIDIA_SMI 可替换为打桩命令）
if [[ "${NM_SHARD_CHECK_GPU_ONLY:-0}" == 1 ]]; then check_local_gpu; exit $?; fi
# Vulkan 静态 TLS 纵深防御（原脚本同款，只给客户端）
TLS_TUNABLE="glibc.rtld.optional_static_tls=65536"

[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$SEED" =~ ^[0-9]+$ && "$SHARD_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$EVAL_TIMEOUT" =~ ^[1-9][0-9]*$ && "$CHUNK_EPISODES" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$POLICY" =~ ^[a-zA-Z0-9_.-]+$ && "$POLICY_CONFIG" =~ ^[a-zA-Z0-9_]+$ && "$EXPECT_CKPT_ID" =~ ^[0-9]+$ && "$SHARD_INDEX" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ -z "$EPISODE_WALL_S" || "$EPISODE_WALL_S" =~ ^[1-9][0-9]*$ ]] || exit 2
DEMO_PREFIX_STORE="${DEMO_PREFIX_STORE:-}"
if [[ -n "$DEMO_PREFIX_STORE" ]]; then
    [[ "$DEMO_PREFIX_STORE" == /* ]] || { echo "DEMO_PREFIX_STORE 必须是绝对路径：$DEMO_PREFIX_STORE"; exit 2; }
    [[ -f "$DEMO_PREFIX_STORE/index.json" ]] || { echo "demo 前缀库缺 index.json：$DEMO_PREFIX_STORE"; exit 2; }
fi
[[ -z "${EXPECT_DEMO_INJECT:-}" || "$EXPECT_DEMO_INJECT" =~ ^[01]$ ]] || { echo 'EXPECT_DEMO_INJECT 只能是 0 / 1'; exit 2; }
[[ -z "${EXPECT_DEMO_INJECT:-}" ]] || export EXPECT_DEMO_INJECT
# NativeMEM 没有 motion 记忆：check_shard.py 的 MOTION_WINDOWS 闸只能 SKIP
export EXPECT_MOTION_STATS=0
unset MMEVLA_MOTION_OVERFLOW MMEVLA_MOTION_PROV_RELAX MMEVLA_ENC_CHUNK MMEVLA_MOTION_ONLINE_GPU
[[ "$CKPT" == /* ]] || { echo "checkpoint 必须是绝对路径：$CKPT"; exit 2; }
[[ "$DRY" == 1 || ( -d "$CKPT/params" && -f "$CKPT/assets/robomme4task/norm_stats.json" ) ]] \
    || { echo "checkpoint 缺 params/ 或 assets/robomme4task/norm_stats.json：$CKPT"; exit 2; }
[[ "$DRY" == 1 || -f "$PLAN" ]] || { echo "分片计划不存在：$PLAN"; exit 2; }
CKPT_ID="$(basename "$CKPT")"
[[ "$CKPT_ID" == "$EXPECT_CKPT_ID" ]] || { echo "本轮只验收 step $EXPECT_CKPT_ID，得到 $CKPT_ID"; exit 2; }
RUN_DIR="$NM_EVAL_ROOT/$RUN_NAME/$SHARD_TAG"
RESULT_ROOT="$RUN_DIR/$POLICY/ckpt$CKPT_ID/seed$SEED"
SERVER_PY="$NM_SRC/scripts/robomme/serve_nativemem.py"
CLIENT_PY="$NM_SRC/scripts/robomme/client/eval.py"
nm_assert_writable RUN_DIR "$RUN_DIR" "$RESULT_ROOT" || exit 2
if [[ "$DRY" == 1 ]]; then
    nm_env_report || exit 2
    echo "RUN_DIR=$RUN_DIR"
    echo "RESULT_ROOT=$RESULT_ROOT"
    echo "SERVER_PY=$SERVER_PY"
    echo "CLIENT_PY=$CLIENT_PY"
    echo "PLAN=$PLAN CKPT=$CKPT CKPT_ID=$CKPT_ID SEED=$SEED POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG obs_horizon=$OBS_HORIZON"
    echo "DRY_RUN_SHARD=PASS writable_in_policy=0"
    exit 0
fi
if [[ -e "$RESULT_ROOT/progress.json" && "${ALLOW_RESUME:-0}" != 1 ]]; then
    echo "运行结果已存在：$RESULT_ROOT/progress.json（要续跑请显式 ALLOW_RESUME=1）"; exit 2
fi
# clean 检查：policy 仓只读确认；fork 排除 git 忽略不到的产物目录 runs/
# --no-optional-locks：git status 默认会顺手刷新 .git/index，对 policy 仓必须真正只读
[[ -z "$(git --no-optional-locks -C "$POLICY_SRC" status --porcelain)" ]] || { echo 'policy 仓工作区不干净，停止（只读确认）'; exit 2; }
[[ -z "$(git -C "$NM_SRC" status --porcelain -- . ':(exclude)runs')" ]] || { echo 'fork 必须从 clean HEAD 启动'; exit 2; }
command -v uv
check_local_gpu
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
SERVER_PID=''
collect_tree() {
    local pid="$1" child kids
    printf '%s\n' "$pid"
    kids="$(cat "/proc/$pid/task/$pid/children" 2>/dev/null)"
    [[ -n "$kids" ]] || kids="$(ps -o pid= --ppid "$pid" 2>/dev/null)"
    for child in $kids; do
        collect_tree "$child"
    done
}
cleanup() {
    RC=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        local P
        for P in $(collect_tree "$SERVER_PID" | tac); do
            kill "$P" 2>/dev/null || true
        done
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    date -u +END_UTC=%Y-%m-%dT%H:%M:%SZ
    echo "EXIT_CODE=$RC"
    exit "$RC"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u +START_UTC=%Y-%m-%dT%H:%M:%SZ
hostname
echo "NM_SRC_HEAD=$(git -C "$NM_SRC" rev-parse HEAD) POLICY_SRC_HEAD=$(git -C "$POLICY_SRC" rev-parse HEAD)"
uv --version
nvidia-smi --query-gpu=name,uuid,memory.used,compute_mode --format=csv
printf 'NM_SRC=%s POLICY_SRC=%s SHARD=%s CUDA_VISIBLE_DEVICES=%s RUN_DIR=%s\n' \
    "$NM_SRC" "$POLICY_SRC" "$SHARD_TAG" "${CUDA_VISIBLE_DEVICES:-}" "$RUN_DIR"
[[ "${ROBOMME_GPU_RASTER:-0}" == 0 && "${SAPIEN_DISABLE_RAY_TRACING:-0}" == 0 ]] || { echo '本轮仅允许原生渲染'; exit 2; }
PLAN_EPISODES="$("${CLIENT_UV[@]}" -c "
import json,sys
plan=json.load(open(sys.argv[1]))
print(sum(len(v) for v in plan['groups'].values()))
" "$PLAN")"
echo "PLAN=$PLAN episodes=$PLAN_EPISODES chunk=$CHUNK_EPISODES"
"${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/verify_sources.py" | tee "$RUN_DIR/sources.log"
PROBE_TASK="$("${CLIENT_UV[@]}" -c "
import json,sys
plan=json.load(open(sys.argv[1]))
print(sorted(plan['groups'])[0].split('/')[0])
" "$PLAN")"
timeout --kill-after=15s 180s "${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/render_probe.py" "$RUN_DIR/render" "$PROBE_TASK"

PORT_SLOT="$SHARD_INDEX"; [[ "$PORT_SLOT" =~ ^[0-9]+$ ]] || PORT_SLOT=$(( $(printf '%s' "$PORT_SLOT" | cksum | cut -d' ' -f1) % 100 ))
PORT=$((18011 + (PORT_SLOT % 100) * 7))
while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
    PORT=$((PORT + 1))
    (( PORT < 18811 )) || exit 2
done
check_local_gpu
echo "EVAL_PARAMS policy=$POLICY config=$POLICY_CONFIG ckpt=$CKPT_ID seed=$SEED max_steps=$MAX_STEPS obs_horizon=$OBS_HORIZON wall=${EPISODE_WALL_S:-default}"
echo "EVAL_PARAMS_DEMO demo_prefix_store=${DEMO_PREFIX_STORE:-none} expect_demo_inject=${EXPECT_DEMO_INJECT:-0}"
"${SERVER_UV[@]}" "$SERVER_PY" --ckpt "$CKPT" --config "$POLICY_CONFIG" --seed "$SEED" --port "$PORT" \
    > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for ((ATTEMPT=0; ATTEMPT<240; ATTEMPT++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 80 "$RUN_DIR/server.log"; exit 1; }
    if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then READY=1; break; fi
    sleep 3
done
[[ "$READY" == 1 ]] || { tail -n 80 "$RUN_DIR/server.log"; exit 1; }
# 端口归属：监听者必须在本次 server 进程树内（原脚本同款）
HEXPORT=$(printf '%04X' "$PORT")
SERVER_TREE="$(collect_tree "$SERVER_PID" | tr '\n' ' ')"
OWN=0
for INODE in $(awk -v p=":$HEXPORT" '$2 ~ p"$" && $4=="0A" {print $10}' /proc/net/tcp /proc/net/tcp6 2>/dev/null); do
    for P in $SERVER_TREE; do
        if ls -l "/proc/$P/fd" 2>/dev/null | grep -q "socket:\[$INODE\]"; then OWN=1; break 2; fi
    done
done
[[ "$OWN" == 1 ]] || {
    echo "端口 $PORT 的监听者不在本次 server 进程树（根 PID=$SERVER_PID，树=$SERVER_TREE），停止"
    exit 2
}
grep -E '^(NORM_SHA|SERVE_RNG|ENCODER_WARMUP|SERVE_NATIVEMEM)' "$RUN_DIR/server.log" || true
echo "SERVER_READY port=$PORT pid=$SERVER_PID own=1 tree=$SERVER_TREE"

CHUNKS=$(( (PLAN_EPISODES + CHUNK_EPISODES - 1) / CHUNK_EPISODES ))
for ((CHUNK=1; CHUNK<=CHUNKS; CHUNK++)); do
    echo "CHUNK_START $CHUNK/$CHUNKS"
    timeout --signal=TERM --kill-after=20s "${EVAL_TIMEOUT}s" \
        env "GLIBC_TUNABLES=$TLS_TUNABLE" "${CLIENT_UV[@]}" "$CLIENT_PY" \
        --args.host=127.0.0.1 --args.port="$PORT" --args.model_seed="$SEED" \
        --args.obs_horizon="$OBS_HORIZON" \
        --args.policy_name="$POLICY" --args.model_ckpt_id="$CKPT_ID" \
        --args.max_steps="$MAX_STEPS" --args.save_dir="$RUN_DIR" \
        --args.episode_plan="$PLAN" --args.shard_tag="$SHARD_TAG" \
        --args.max_new_episodes="$CHUNK_EPISODES" \
        ${EPISODE_WALL_S:+--args.episode_wall_s="$EPISODE_WALL_S"} \
        ${DEMO_PREFIX_STORE:+--args.demo_prefix_store="$DEMO_PREFIX_STORE"} \
        2>&1 | tee -a "$RUN_DIR/client.log"
    echo "CHUNK_DONE $CHUNK/$CHUNKS"
done

"${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/check_shard.py" "$RUN_DIR" "$POLICY" "$CKPT_ID" "$SEED" "$PLAN"
echo "SHARD_DONE run=$RUN_NAME shard=$SHARD_TAG episodes=$PLAN_EPISODES"
