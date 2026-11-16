#!/usr/bin/env bash
# NativeMEM 评估环境（1116 计划 C2 第 3 行、J#6 双根路径）。只被 source，不单独执行。
# 两个根：
#   POLICY_SRC  policy 仓，只读：评估客户端解释器与其 venv、robomme 候选库、demo 前缀库、分片计划与合并脚本；
#               本文件不 source policy 仓的 scripts/evaluation/env.sh，也不把任何可写路径放进 policy 仓。
#   NM_ROOT     fork 主检出 /scratch/hongze/NativeMEM：一切评估输出、日志、gpu-slots 落 $NM_ROOT/runs/eval/<run>。
# NM_SRC 是本文件所在的 fork 源码根（主检出或 worktree）：服务器、客户端副本都从这里取。
NM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_SRC="${POLICY_SRC:-/scratch/hongze/robomme_policy_learning_MotionJEPA}"
NM_ROOT="${NM_ROOT:-/scratch/hongze/NativeMEM}"
NM_EVAL_ROOT="$NM_ROOT/runs/eval"
NM_VENV="${NM_VENV:-$NM_ROOT/.venv}"
NM_CACHE="/scratch/hongze/.cache"
CLIENT_ENV="$POLICY_SRC/v1-store/envs/policy-eval-client"
CLIENT_PROJECT="$POLICY_SRC/scripts/evaluation/client"
POLICY_EVAL_SCRIPTS="$POLICY_SRC/scripts/evaluation"
CANDIDATES="$POLICY_SRC/third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"
export NM_SRC POLICY_SRC NM_ROOT NM_EVAL_ROOT NM_VENV

unset VIRTUAL_ENV
# 缓存全部落 /scratch/hongze/.cache（全局规则）；不指向 policy 仓 v1-store/cache
export UV_LINK_MODE=copy UV_CACHE_DIR="$NM_CACHE/uv"
export XDG_CACHE_HOME="$NM_CACHE/nm-eval/xdg"
export HF_HOME="$NM_CACHE/nm-eval/huggingface"
export OPENPI_DATA_HOME="$NM_CACHE/openpi"
export TORCH_HOME="$NM_CACHE/nm-eval/torch"
export MS_ASSET_DIR="$NM_CACHE/nm-eval/maniskill"
export CUDA_CACHE_PATH="$NM_CACHE/nm-eval/cuda"
export JAX_COMPILATION_CACHE_DIR="$NM_CACHE/nm-eval/jax"
export WANDB_DIR="$NM_CACHE/nm-eval/wandb" WANDB_MODE=disabled
# 两个解释器都会 import policy 仓源码（客户端的 robomme / openpi_client 是指向 policy 仓的 editable）：不写 __pycache__
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
# 服务端：fork 的 .venv；PYTHONPATH 指向 NM_SRC/src，保证 worktree 里跑时 import 的是 worktree 代码
SERVER_UV=(env "UV_PROJECT_ENVIRONMENT=$NM_VENV" "PYTHONPATH=$NM_SRC/src" uv run --project "$NM_SRC" --frozen --no-sync python)
# 客户端：policy 仓的独立评估 venv（只读使用，--frozen --no-sync 不改 lock、不装包）
CLIENT_UV=(env -u PYTHONPATH "UV_PROJECT_ENVIRONMENT=$CLIENT_ENV" uv run --project "$CLIENT_PROJECT" --frozen --no-sync python)
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi

# 断言可写路径不落在 policy 仓：nm_assert_writable <名字> <路径>...（路径不存在也按字面与上级 realpath 判）
nm_assert_writable() {
    local name="$1" p real policy_real
    shift
    policy_real="$(realpath -m "$POLICY_SRC")"
    for p in "$@"; do
        [[ "$p" == /* ]] || { echo "WRITABLE_PATH=FAIL name=$name reason=not-absolute path=$p"; return 2; }
        real="$(realpath -m "$p")"
        if [[ "$real" == "$policy_real" || "$real" == "$policy_real"/* ]]; then
            echo "WRITABLE_PATH=FAIL name=$name path=$p real=$real（落在 policy 仓）"
            return 2
        fi
    done
}

# 打印并断言 eval_env 本身的全部路径（DRY_RUN 用）
nm_env_report() {
    echo "NM_SRC=$NM_SRC"
    echo "NM_ROOT=$NM_ROOT"
    echo "NM_EVAL_ROOT=$NM_EVAL_ROOT"
    echo "NM_VENV=$NM_VENV"
    echo "POLICY_SRC=$POLICY_SRC (只读)"
    echo "CLIENT_ENV=$CLIENT_ENV (只读)"
    echo "CLIENT_PROJECT=$CLIENT_PROJECT (只读)"
    echo "POLICY_EVAL_SCRIPTS=$POLICY_EVAL_SCRIPTS (只读)"
    echo "CANDIDATES=$CANDIDATES (只读)"
    echo "SERVER_UV=${SERVER_UV[*]}"
    echo "CLIENT_UV=${CLIENT_UV[*]}"
    local v
    for v in UV_CACHE_DIR XDG_CACHE_HOME HF_HOME OPENPI_DATA_HOME TORCH_HOME MS_ASSET_DIR CUDA_CACHE_PATH \
             JAX_COMPILATION_CACHE_DIR WANDB_DIR; do
        echo "CACHE $v=${!v}"
        nm_assert_writable "$v" "${!v}" || return 2
    done
    nm_assert_writable NM_EVAL_ROOT "$NM_EVAL_ROOT" || return 2
}
