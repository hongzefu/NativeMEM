#!/usr/bin/env bash
# 把正式 run nativemem-4task1600-b128-20k 的最终存档（只有 19999 一份）整目录传到 HF 公开 bucket
# HongzeFu/robomme-vla-4task-nativemem-20k-v1，并在本机读回逐文件 sha256／字节／对象数对照（1116 计划 C1、G13）。
#
# 由 policy 仓 scripts/dataset/hf_export/run_ctxmotion160_ckpt_export.sh 的做法改写（骨架照抄）：
#   硬链接 stage、不打 tar、uvx 拉独立 huggingface_hub[cli]==1.30.0、上传前 / 读回后两遍独立 sha256 逐行 diff、
#   源侧第三遍 sha256 证明硬链接未改原件、建公开 bucket（不带 --private）+ 空库断言（uploaded.marker 区分续传）。
# 与蓝本的差异：
#   1. 只有一份存档：step 19999；前置要求 G12 判定文件 logs/train/<run>.TRAIN_FINAL.verdict 为 TRAIN_FINAL=PASS，
#      run 目录下恰有一个数字目录 19999、无 orbax tmp 目录、19999/_CHECKPOINT_METADATA 存在；
#   2. 凭据经 set -a; . /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/secrets/hf.env 读取（只读引用，不打印）；
#   3. stage／verify／日志落 fork 的 git 忽略目录 checkpoints/_exports/ 与 logs/；
#   4. 末行 HF_VERIFY=PASS ckpts=1 files=<n> bytes=<n>（另核对 bucket 对象数 == 本地文件数）。
# 用法：bash scripts/robomme/upload_ckpt.sh（放 detached tmux nm-ckptup；可整脚本重跑续传）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/scripts/train_nativemem.py" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

CONFIG=nativemem_pi05_robomme4task
RUN_NAME="${RUN_NAME:-nativemem-4task1600-b128-20k}"
STEP=19999
BUCKET_ID="HongzeFu/robomme-vla-4task-nativemem-20k-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="${SRC:-$REPO/checkpoints/$CONFIG/$RUN_NAME}"
VERDICT="${VERDICT:-$REPO/logs/train/$RUN_NAME.TRAIN_FINAL.verdict}"
SECRETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/secrets/hf.env
EXPORT_ROOT="$REPO/checkpoints/_exports/hf-ckpt-$RUN_NAME"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
LOGD="$REPO/logs/ckptup-$RUN_NAME"
NPROC=48

# 缓存逐项指向 /scratch/hongze/.cache（禁止覆盖 HOME）
export HF_HOME=/scratch/hongze/.cache/hf-upload-nativemem
export HF_XET_CACHE=/scratch/hongze/.cache/hf-upload-nativemem/xet
export HF_XET_HIGH_PERFORMANCE=1
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# 凭据：只从 policy 仓 v1-store/secrets/hf.env 取，不打印；环境里原有的 HF_TOKEN 不属于 HongzeFu，必须被覆盖。
[ -f "$SECRETS" ] || { echo "错误: 缺少 $SECRETS"; exit 1; }
set -a; . "$SECRETS"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

bucket_stat() {  # $1 = 字段名 size|totalFiles
  curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/buckets/HongzeFu" \
    | uv run --no-project python -c "
import sys, json
for b in json.load(sys.stdin):
    if b['id'] == '$BUCKET_ID':
        print(b.get('$1')); break
else:
    sys.exit('目标 bucket $BUCKET_ID 不在返回列表里')
"
}

hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）；按路径排序
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

retry_sync() {  # $1 = 源, $2 = 目标；commit 超时是间歇性的，重试 8 次
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$1" "$2"; then return 0; fi
    echo "  sync [$1 → $2] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  echo "错误: sync [$1 → $2] 八次仍失败"; exit 1
}

echo "阶段0开始：前置（G12）+ 预检 + 建 bucket $(TZ=America/New_York date '+%F %T 美东')"
[ -f "$VERDICT" ] || { echo "错误: 缺 G12 判定文件 $VERDICT"; exit 1; }
grep -q "^TRAIN_FINAL=PASS step=$STEP ckpts=1 finite=PASS" "$VERDICT" || { echo "错误: G12 未 PASS：$(cat "$VERDICT")"; exit 1; }
echo "  G12: $(cat "$VERDICT")"
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
dirs=$( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort | tr '\n' ' ' )
[ "$dirs" = "$STEP " ] || { echo "错误: run 目录下应只有 $STEP（实际: $dirs）"; exit 1; }
[ -f "$SRC/$STEP/_CHECKPOINT_METADATA" ] || { echo "错误: $SRC/$STEP/_CHECKPOINT_METADATA 不存在（未落盘完成）"; exit 1; }
for sub in params train_state assets; do
  [ -d "$SRC/$STEP/$sub" ] || { echo "错误: 存档缺 $sub/"; exit 1; }
done
for d in "$REPO/checkpoints/_exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$LOGD/sums" "$HF_XET_CACHE"
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
case "$who" in *HongzeFu*) echo "HF_WHOAMI=HongzeFu" ;; *) echo "错误: 身份不是 HongzeFu"; exit 1 ;; esac
# 公开 bucket，不带 --private；已存在则 create 非零退出，由空库断言兜底
if hf buckets create "$BUCKET_ID" 2>&1 | tee "$TMPD/create.log"; then
  echo "  bucket create 成功"
else
  echo "  bucket create 非零退出（可能已存在），继续走空库断言"
fi
pre_files="$(bucket_stat totalFiles)"
echo "  目标 bucket 现状: files=$pre_files"
if [ "$pre_files" != "0" ] && [ ! -f "$LOGD/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"; exit 1
fi
touch "$LOGD/uploaded.marker"
echo "阶段0完成"

echo "阶段1开始：stage 硬链接 → 上传前 sha256 → 上传 → 读回 → 对照"
rm -rf "${STAGE:?}/$STEP"; mkdir -p "$STAGE/$STEP"
( cd "$SRC/$STEP" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do mkdir -p "$STAGE/$STEP/$d"; done
( cd "$SRC/$STEP" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do ln -f "$SRC/$STEP/$f" "$STAGE/$STEP/$f"; done
if find "$STAGE/$STEP" -type l -print | grep -q .; then echo "错误: stage 内出现符号链接"; exit 1; fi
hash_tree "$STAGE/$STEP" "$TMPD/pre.txt"
files=$(find "$STAGE/$STEP" -type f | wc -l)
bytes=$(find "$STAGE/$STEP" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
echo "  stage files=$files bytes=$bytes pre_lines=$(wc -l < "$TMPD/pre.txt")"
retry_sync "$STAGE/$STEP" "$BUCKET/$STEP"
rm -rf "${VERIFY:?}/$STEP"; mkdir -p "$VERIFY/$STEP"
retry_sync "$BUCKET/$STEP" "$VERIFY/$STEP"
hash_tree "$VERIFY/$STEP" "$TMPD/post.txt"
diff "$TMPD/pre.txt" "$TMPD/post.txt" || { echo "错误: 上传前后 sha256 不一致（上方 diff）"; echo "HF_VERIFY=FAIL ckpts=0"; exit 1; }
vbytes=$(find "$VERIFY/$STEP" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
[ "$vbytes" = "$bytes" ] || { echo "错误: 读回字节数 $vbytes != $bytes"; echo "HF_VERIFY=FAIL ckpts=0"; exit 1; }
hash_tree "$SRC/$STEP" "$TMPD/src-after.txt"
diff "$TMPD/pre.txt" "$TMPD/src-after.txt" || { echo "错误: 源文件 sha256 发生变化（上方 diff）"; echo "HF_VERIFY=FAIL ckpts=0"; exit 1; }
post_files="$(bucket_stat totalFiles)"
[ "$post_files" = "$files" ] || { echo "错误: bucket 对象数 $post_files != 本地文件数 $files"; echo "HF_VERIFY=FAIL ckpts=0"; exit 1; }
cp "$TMPD/pre.txt" "$LOGD/sums/$STEP.SHA256SUMS.txt"
echo "STEP_VERIFY=PASS sub=$STEP files=$files bytes=$bytes bucket_files=$post_files"
echo "阶段1完成 $(TZ=America/New_York date '+%F %T 美东')"
echo "HF_VERIFY=PASS ckpts=1 files=$files bytes=$bytes"
