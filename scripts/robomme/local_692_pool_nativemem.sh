#!/usr/bin/env bash
# NativeMEM 本机 692 局评估驱动（1116 计划 C2 第 3 行、J#6 #7）。
# 由 policy 仓 scripts/evaluation/local_692_pool.sh 复制改写，流程（E0 闸 → 分片计划 → 预标 9 条 → 每卡 flock 占位
# 同跑 → scan → 新 error 重跑一次 → check_shard → merge → 判定行）保持不变，差异只有：
#   - 双根路径：source 本目录 eval_env.sh（不 source policy 仓 env.sh）；POLICY_SRC 只读，一切输出落
#     $NM_EVAL_ROOT/<RUN_NAME>（/scratch/hongze/NativeMEM/runs/eval/<run>）；policy 仓脚本全用绝对路径调用；
#   - 分片调 run_shard_nativemem.sh（服务器 serve_nativemem.py、客户端副本、执行 48 步）；去掉 motion 变量；
#   - 每卡份数 EVAL_PER_GPU 默认 4（32 片）；=3 时 24 片；硬断言 SHARDS == 8*PER_GPU；
#     片数不是 32 时另生成 32 片参照计划，核对两份 692 身份集合相等（SHARD_PLAN_SAME=PASS n=692，G11a）；
#   - DRY_RUN=1 打印全部解析后的绝对路径，并断言没有一个可写路径落在 policy 仓（J#6）。
#
# 用法：
#   local_692_pool_nativemem.sh <checkpoint 绝对路径> [seed=7]            正式评估（ckpt …/19999）
#   local_692_pool_nativemem.sh --g6b <冒烟 checkpoint 绝对路径> <seed>     评估加载冒烟（G10，ckpt …/nm-smoke20/19）
#   local_692_pool_nativemem.sh --g6b-capacity <checkpoint 绝对路径> <seed> 每卡 N 份显存压测（G11）
# 可覆盖：RUN_NAME（正式默认 local692-nativemem-20k，G6b 默认 nm-g6b，压测默认 nm-cap<EVAL_PER_GPU>）、
#        POLICY（结果目录层级名，默认 nativemem-4task1600-b128-20k）、POLICY_CONFIG（默认 nativemem_pi05_robomme4task）、
#        STORE（demo 前缀库根，默认 $POLICY_SRC/v1-store/demo-prefix/binfill-3b4de03a0b46）、DRY_RUN=1、
#        EVAL_PER_GPU（默认 4；只接受 3 或 4）、EVAL_SHARDS（默认 8*EVAL_PER_GPU，必须相等）、
#        EVAL_MEM_FRACTION（默认 4 份 0.2、3 份 0.28）；G6b 模式保持原行为（10 片计划、单片 0.7、同卡 2 片 0.45）。
# 判定行：EVAL692=PASS episodes=<n> success=<k> rate=<x> preseeded_errors=<n> new_errors=<n> retried=<n>
#        EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=<x>
#        EVAL_CAPACITY=PASS|FAIL per_gpu=<n> peak_mem_gb=<x> free_gb=<y>
set -uo pipefail
source "$(dirname "$0")/eval_env.sh"

DRY="${DRY_RUN:-0}"
MODE=eval
if [[ "${1:-}" == --g6b ]]; then MODE=g6b; shift; elif [[ "${1:-}" == --g6b-capacity ]]; then MODE=cap; shift; fi
CKPT="${1:?需要 checkpoint 绝对路径}"
SEED="${2:-7}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed 必须是非负整数：$SEED"; exit 2; }
[[ "$CKPT" == /* ]] || { echo "checkpoint 必须是绝对路径：$CKPT"; exit 2; }
[[ "$DRY" == 1 || ( -d "$CKPT/params" && -f "$CKPT/assets/robomme4task/norm_stats.json" ) ]] \
    || { echo "checkpoint 缺 params/ 或 assets/robomme4task/norm_stats.json：$CKPT"; exit 2; }
CKPT_ID="$(basename "$CKPT")"
if [[ "$MODE" == g6b ]]; then
    SHARDS=10; PER_GPU=2; MEM_FRACTION=0.45   # G6b 保持原行为，不读 EVAL_* 覆盖
else
    PER_GPU="${EVAL_PER_GPU:-4}"
    [[ "$PER_GPU" == 4 || "$PER_GPU" == 3 ]] || { echo "EVAL_PER_GPU 只接受 4 或 3：$PER_GPU"; exit 2; }
    SHARDS="${EVAL_SHARDS:-$((8 * PER_GPU))}"
    [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_SHARDS 必须是正整数：$SHARDS"; exit 2; }
    (( SHARDS == 8 * PER_GPU )) || { echo "SHARDS=$SHARDS 必须等于 8*PER_GPU=$((8 * PER_GPU))"; exit 2; }
    if [[ "$PER_GPU" == 4 ]]; then MEM_FRACTION="${EVAL_MEM_FRACTION:-0.2}"; else MEM_FRACTION="${EVAL_MEM_FRACTION:-0.28}"; fi
fi
if [[ "$MODE" == g6b ]]; then RUN_NAME="${RUN_NAME:-nm-g6b}"
elif [[ "$MODE" == cap ]]; then RUN_NAME="${RUN_NAME:-nm-cap$PER_GPU}"
else RUN_NAME="${RUN_NAME:-local692-nativemem-20k}"; fi
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "RUN_NAME 非法：$RUN_NAME"; exit 2; }
STORE="${STORE:-$POLICY_SRC/v1-store/demo-prefix/binfill-3b4de03a0b46}"
PRESEED_SRC="$POLICY_SRC/docs/training-doc/finalckpt-nomotion59999-gl/records/preseeded-errors.json"
EVAL_ROOT="$NM_EVAL_ROOT/$RUN_NAME"
RETRY_RUN="$RUN_NAME-retry1"
RETRY_ROOT="$NM_EVAL_ROOT/$RETRY_RUN"
SHARD_SH="$NM_SRC/scripts/robomme/run_shard_nativemem.sh"
SLOT_DIR="$EVAL_ROOT/gpu-slots"

export EXPECT_DEMO_INJECT=1 EXPECT_MOTION_STATS=0 CHUNK_EPISODES=20 EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 \
       LOCAL_GPU_SHARE="$PER_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" ALLOW_RESUME=1 \
       POLICY="${POLICY:-nativemem-4task1600-b128-20k}" POLICY_CONFIG="${POLICY_CONFIG:-nativemem_pi05_robomme4task}" \
       EXPECT_CKPT_ID="$CKPT_ID" DEMO_PREFIX_STORE="$STORE"
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING MMEVLA_MOTION_OVERFLOW MMEVLA_MOTION_PROV_RELAX MMEVLA_ENC_CHUNK
echo "LOCAL692_START mode=$MODE dry=$DRY run=$RUN_NAME ckpt=$CKPT seed=$SEED store=$STORE $(date -u +%FT%TZ)"
echo "EVAL_ENV EXPECT_DEMO_INJECT=$EXPECT_DEMO_INJECT EXPECT_MOTION_STATS=$EXPECT_MOTION_STATS CHUNK_EPISODES=$CHUNK_EPISODES" \
     "seed=$SEED max_steps=2000 obs_horizon=48(run_shard_nativemem.sh) EPISODE_WALL_S=$EPISODE_WALL_S EVAL_TIMEOUT=$EVAL_TIMEOUT" \
     "LOCAL_GPU_SHARE=$LOCAL_GPU_SHARE XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION POLICY=$POLICY" \
     "POLICY_CONFIG=$POLICY_CONFIG EXPECT_CKPT_ID=$EXPECT_CKPT_ID SHARDS=$SHARDS PER_GPU=$PER_GPU MEM_FRACTION=$MEM_FRACTION"

# ── 可写路径断言（J#6）：DRY 与正式都先过这一道 ──
nm_assert_writable EVAL_ROOT "$EVAL_ROOT" "$RETRY_ROOT" "$SLOT_DIR" "$EVAL_ROOT/plans" "$EVAL_ROOT/logs" || exit 2
if [[ "$DRY" == 1 ]]; then
    nm_env_report || exit 2
    echo "EVAL_ROOT=$EVAL_ROOT"
    echo "RETRY_ROOT=$RETRY_ROOT"
    echo "SLOT_DIR=$SLOT_DIR"
    echo "SHARD_SH=$SHARD_SH"
    echo "STORE=$STORE (只读)"
    echo "PRESEED_SRC=$PRESEED_SRC (只读)"
    echo "CKPT=$CKPT (只读)"
fi

# ── 内嵌 Python（经 CLIENT_UV 运行；只用标准库）──
read -r -d '' PYHELP <<'PY'
import json, shutil, sys
from pathlib import Path

def plan_groups(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["groups"]

def result_root(shard_dir, policy, ckpt, seed):
    return Path(shard_dir) / policy / f"ckpt{ckpt}" / f"seed{seed}"

def preseed_keys(src):
    # 只取身份：各 run 的 by_shard 并集去重，不复用分片号（C5）
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    keys = set()
    for run in data.values():
        for groups in run["by_shard"].values():
            for group, eps in groups.items():
                keys |= {(group, int(e)) for e in eps}
    return sorted(keys)

def cmd_preseed(plan_dir, src, out_root, policy, ckpt, seed, shards):
    keys = preseed_keys(src)
    owner = {}
    for k in range(int(shards)):
        for group, eps in plan_groups(Path(plan_dir) / f"shard{k}.json").items():
            for e in eps:
                owner.setdefault((group, int(e)), []).append(k)
    placed, dup, by_shard = 0, 0, {}
    for key in keys:
        hits = owner.get(key, [])
        if len(hits) > 1:
            dup += 1
        if len(hits) == 1:
            placed += 1
            by_shard.setdefault(f"s{hits[0]}", {}).setdefault(key[0], []).append(str(key[1]))
        print(f"PRESEED_PLACE key={key[0]}/{key[1]} shards={','.join(f's{h}' for h in hits) or 'none'}", flush=True)
    ok = len(keys) == 9 and placed == 9 and dup == 0
    line = f"PRESEED={'PASS' if ok else 'FAIL'} episodes={len(keys)} placed={placed} duplicates={dup}"
    if ok and out_root != "-":
        for tag, groups in by_shard.items():
            root = result_root(Path(out_root) / tag, policy, ckpt, seed)
            root.mkdir(parents=True, exist_ok=True)
            path = root / "progress.json"
            if path.exists():
                sys.exit(f"预标目标已存在，拒绝覆盖：{path}")
            path.write_text(json.dumps({g: {e: "error" for e in eps} for g, eps in groups.items()}, indent=2),
                            encoding="utf-8")
        record = {"source": str(src), "note": f"身份引自历史清单，按本次 {shards} 片计划重定位，未复用其分片号",
                  "keys": [f"{g}/{e}" for g, e in keys], "by_shard": by_shard, "verdict": line}
        (Path(out_root) / "preseeded-errors.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
    print(line, flush=True)
    sys.exit(0 if ok else 1)

def cmd_scan(eval_root, plan_dir, policy, ckpt, seed, shards):
    """逐片判定：ok / expected（只含预标 error）/ retry（含新 error，写 retry1-plan.json）/ infra（不可重试）。"""
    rec = json.loads((Path(eval_root) / "preseeded-errors.json").read_text(encoding="utf-8"))
    pre = set(rec["keys"])
    for k in range(int(shards)):
        tag = f"s{k}"
        rc = int((Path(eval_root) / "logs" / f"{tag}.rc").read_text().strip())
        plan = json.loads((Path(plan_dir) / f"shard{k}.json").read_text(encoding="utf-8"))
        planned = {(g, int(e)) for g, eps in plan["groups"].items() for e in eps}
        path = result_root(Path(eval_root) / tag, policy, ckpt, seed) / "progress.json"
        progress = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        done = {(g, int(e)) for g, entries in progress.items() for e in entries}
        errors = sorted(f"{g}/{e}" for g, entries in progress.items() for e, v in entries.items() if v == "error")
        new = [key for key in errors if key not in pre]
        complete = done == planned
        if not complete or (rc != 0 and not errors):
            verdict = "infra"
        elif new:
            verdict = "retry"
            groups = {}
            for key in new:
                task, diff, ep = key.split("/")
                groups.setdefault(f"{task}/{diff}", []).append(int(ep))
            retry = {"shard_tag": f"{tag}-retry1", "candidates": plan["candidates"],
                     "identity_sha256": plan["identity_sha256"],
                     "groups": {g: sorted(v) for g, v in sorted(groups.items())}}
            (Path(eval_root) / tag / "retry1-plan.json").write_text(
                json.dumps(retry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            verdict = "expected" if rc != 0 else "ok"
        print(f"SHARD_SCAN shard={tag} rc={rc} complete={int(complete)} errors={len(errors)} "
              f"preseeded={len(errors) - len(new)} new_errors={len(new)} verdict={verdict} new={','.join(new) or '-'}",
              flush=True)

def _videos_of(videos_dir, key):
    task, diff, ep = key.split("/")
    return sorted(p for p in Path(videos_dir).glob(f"{task}_ep{ep}_*_{diff}.mp4"))

def cmd_integrate(eval_root, retry_root, policy, ckpt, seed, shards):
    """C7：把 retry1 的结果并回最终目录；首次尝试的该局产物移入 attempt0/。"""
    records = []
    for k in range(int(shards)):
        tag = f"s{k}"
        shard_dir = Path(eval_root) / tag
        plan_path = shard_dir / "retry1-plan.json"
        if not plan_path.is_file():
            continue
        keys = [f"{g}/{e}" for g, eps in plan_groups(plan_path).items() for e in eps]
        first_rc = int((Path(eval_root) / "logs" / f"{tag}.rc").read_text().strip())
        retry_rc = int((Path(eval_root) / "logs" / f"{tag}-retry1.rc").read_text().strip())
        final = result_root(shard_dir, policy, ckpt, seed)
        retry_dir = Path(retry_root) / tag
        rres = result_root(retry_dir, policy, ckpt, seed)
        rpath = rres / "progress.json"
        rprog = json.loads(rpath.read_text(encoding="utf-8")) if rpath.is_file() else {}
        rvals = {f"{g}/{e}": v for g, entries in rprog.items() for e, v in entries.items()}
        if set(rvals) != set(keys):
            print(f"RETRY_INFRA shard={tag} rc={retry_rc} 重跑目录不完整：期望 {keys} 实得 {sorted(rvals)}", flush=True)
            sys.exit(1)
        attempt0 = shard_dir / "attempt0"
        (attempt0 / "videos").mkdir(parents=True, exist_ok=False)
        prog = json.loads((final / "progress.json").read_text(encoding="utf-8"))
        shutil.copy2(final / "progress.json", attempt0 / "progress.json")
        mpath = final / "motion_stats.json"
        mstats = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
        if mpath.is_file():
            shutil.copy2(mpath, attempt0 / "motion_stats.json")
        if (shard_dir / "client.log").is_file():
            shutil.copy2(shard_dir / "client.log", attempt0 / "client.log")
        rm_path = rres / "motion_stats.json"
        rmstats = json.loads(rm_path.read_text(encoding="utf-8")) if rm_path.is_file() else {}
        rlog = retry_dir / "client.log"
        rlines = rlog.read_text(encoding="utf-8", errors="replace").splitlines() if rlog.is_file() else []
        appended = []
        for key in keys:
            group, ep = key.rsplit("/", 1)
            for v in _videos_of(final / "videos", key):
                shutil.move(str(v), str(attempt0 / "videos" / v.name))
            value = rvals[key]
            prog[group][ep] = value
            if value != "error":
                vids = _videos_of(rres / "videos", key)
                if len(vids) != 1:
                    print(f"RETRY_INFRA shard={tag} key={key} 重跑结果非 error 但视频数={len(vids)}", flush=True)
                    sys.exit(1)
                shutil.copy2(vids[0], final / "videos" / vids[0].name)
            if key in rmstats:
                mstats[key] = rmstats[key]
            else:
                mstats.pop(key, None)
            appended += [line for line in rlines if line.startswith(f"DEMO_PREFIX_INJECTED key={key} ")
                         or line == f"DEMO_PREFIX_INJECTED key={key}"]
            records.append({"shard": tag, "key": key, "first_shard_exit_code": first_rc,
                            "retry_shard_exit_code": retry_rc, "attempts": 2, "final": value})
            print(f"RETRY_MERGED shard={tag} key={key} first=error final={value}", flush=True)
        (final / "progress.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")
        if mstats or mpath.is_file():
            (final / "motion_stats.json").write_text(json.dumps(mstats, indent=2), encoding="utf-8")
        if appended:
            # check_shard.py 的 DEMO_INJECT 闸逐行读 client.log：把重跑那次的注入行带上来源标记追加
            with open(shard_dir / "client.log", "a", encoding="utf-8") as f:
                f.write(f"# RETRY1_MERGED from {retry_dir}/client.log\n")
                f.write("\n".join(appended) + "\n")
        shutil.move(str(retry_dir), str(shard_dir / "retry1"))
    (Path(eval_root) / "retries.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(f"RETRY_INTEGRATE=PASS retried={len(records)}", flush=True)

def cmd_summary(eval_root):
    log = json.loads((Path(eval_root) / "merged" / "log.json").read_text(encoding="utf-8"))
    errors = json.loads((Path(eval_root) / "merged" / "errors.json").read_text(encoding="utf-8"))
    pre = set(json.loads((Path(eval_root) / "preseeded-errors.json").read_text(encoding="utf-8"))["keys"])
    retries = json.loads((Path(eval_root) / "retries.json").read_text(encoding="utf-8"))
    err_keys = {f"{e['group']}/{e['episode']}" for e in errors}
    n, k = int(log["episodes"]), int(log["successes"])
    print(f"EVAL692=PASS episodes={n} success={k} rate={k / n:.4f} preseeded_errors={len(err_keys & pre)} "
          f"new_errors={len(err_keys - pre)} retried={len(retries)}", flush=True)

def cmd_g6b_plans(plan_dir, out_dir):
    # G6b：shard0 / shard1 各取第一条 BinFill（走 demo 注入与 motion demo 窗），各写 1 局计划
    for k, names in ((0, ("g6b-single.json", "g6b-a.json")), (1, ("g6b-b.json",))):
        plan = json.loads((Path(plan_dir) / f"shard{k}.json").read_text(encoding="utf-8"))
        group = sorted(g for g in plan["groups"] if g.startswith("BinFill/"))[0]
        ep = plan["groups"][group][0]
        for name in names:
            one = {"shard_tag": name[:-5], "candidates": plan["candidates"],
                   "identity_sha256": plan["identity_sha256"], "groups": {group: [ep]}}
            (Path(out_dir) / name).write_text(json.dumps(one, indent=2) + "\n", encoding="utf-8")
            print(f"G6B_PLAN file={name} episode={group}/{ep}", flush=True)

def cmd_cap_plans(plan_dir, out_dir, n):
    # 压测：第 k 份取分片 k 的第一局（有 BinFill 优先取 BinFill，走 demo 注入与 motion demo 窗），各写 1 局计划
    for k in range(int(n)):
        plan = json.loads((Path(plan_dir) / f"shard{k}.json").read_text(encoding="utf-8"))
        groups = sorted(g for g in plan["groups"] if g.startswith("BinFill/")) or sorted(plan["groups"])
        group = groups[0]
        ep = plan["groups"][group][0]
        one = {"shard_tag": f"cap{k}", "candidates": plan["candidates"],
               "identity_sha256": plan["identity_sha256"], "groups": {group: [ep]}}
        (Path(out_dir) / f"cap{k}.json").write_text(json.dumps(one, indent=2) + "\n", encoding="utf-8")
        print(f"CAP_PLAN file=cap{k}.json episode={group}/{ep}", flush=True)

def cmd_cap_mem(csv_path):
    # 每行「memory.used, memory.total」（MiB）；打印「峰值GB 余量GB」，无有效采样打印「nan nan」
    rows = []
    for line in Path(csv_path).read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and all(x.isdigit() for x in parts):
            rows.append((int(parts[0]), int(parts[1])))
    if not rows:
        print("nan nan")
        return
    peak = max(u for u, _ in rows)
    total = min(t for _, t in rows)
    print(f"{peak / 1024:.1f} {(total - peak) / 1024:.1f}")

def cmd_episode_ok(shard_dir, policy, ckpt, seed):
    path = result_root(shard_dir, policy, ckpt, seed) / "progress.json"
    prog = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    vals = [v for entries in prog.values() for v in entries.values()]
    print(1 if len(vals) == 1 and type(vals[0]) is bool else 0)

def cmd_peak(csv_path):
    vals = [float(x) for x in Path(csv_path).read_text().split() if x.replace(".", "", 1).isdigit()]
    print(f"{max(vals) / 1024:.1f}" if vals else "nan")

def cmd_plan_same(dir_a, n_a, dir_b, n_b):
    # G11a（J#7）：两份分片计划的 (task/difficulty, episode) 身份集合逐条相等，且各自分片互不相交
    def identities(d, n):
        seen, dup = set(), 0
        for k in range(int(n)):
            for group, eps in plan_groups(Path(d) / f"shard{k}.json").items():
                for e in eps:
                    key = (group, int(e))
                    dup += key in seen
                    seen.add(key)
        return seen, dup
    a, dup_a = identities(dir_a, n_a)
    b, dup_b = identities(dir_b, n_b)
    ok = a == b and dup_a == 0 and dup_b == 0
    print(f"SHARD_PLAN_SAME={'PASS' if ok else 'FAIL'} n={len(b)} shards_a={n_a} shards_b={n_b} "
          f"only_a={len(a - b)} only_b={len(b - a)} dup_a={dup_a} dup_b={dup_b}", flush=True)
    sys.exit(0 if ok else 1)

cmds = {"preseed": cmd_preseed, "scan": cmd_scan, "integrate": cmd_integrate, "summary": cmd_summary,
        "g6b_plans": cmd_g6b_plans, "episode_ok": cmd_episode_ok, "peak": cmd_peak,
        "cap_plans": cmd_cap_plans, "cap_mem": cmd_cap_mem, "plan_same": cmd_plan_same}
cmds[sys.argv[1]](*sys.argv[2:])
PY
py() { "${CLIENT_UV[@]}" -c "$PYHELP" "$@"; }

# ── 每卡占位（C4，原脚本同款）──
slot_reserve() {
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1
      touch "$f"
      n="$(grep -c . "$f")"
      (( n < PER_GPU )) || { echo "SLOT_REFUSED gpu=$1 tag=$2 registered=$n"; exit 3; }
      echo "$2 pending" >> "$f" ) 9>"$f.lock"
}
slot_set_pid() {
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1; sed -i "s/^$2 pending\$/$2 $3/" "$f" ) 9>"$f.lock"
}
slot_release() {
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1; sed -i "/^$2 /d" "$f" ) 9>"$f.lock"
}
run_slot() {  # $1=gpu $2=tag $3=shard_index $4=plan $5=run_name $6=log $7=rc 文件
    local gpu="$1" tag="$2" idx="$3" plan="$4" run="$5" log="$6" rcf="$7" pid rc
    if ! slot_reserve "$gpu" "$tag"; then echo 97 > "$rcf"; return; fi
    env CUDA_VISIBLE_DEVICES="$gpu" SHARD_INDEX="$idx" LOCAL_GPU_SLOT_FILE="$SLOT_DIR/gpu$gpu" \
        bash "$SHARD_SH" "$run" "$CKPT" "$plan" "$SEED" "$tag" > "$log" 2>&1 &
    pid=$!
    slot_set_pid "$gpu" "$tag" "$pid"
    echo "SLOT_REGISTERED gpu=$gpu tag=$tag pid=$pid run=$run"
    wait "$pid"; rc=$?
    slot_release "$gpu" "$tag"
    echo "$rc" > "$rcf"
}

# ── ① E0 闸 ──
E0_LINE="$(head -n 1 "$STORE/E0.txt" 2>/dev/null)"
if [[ "$E0_LINE" != "DEMO_PREFIX_SRC=PASS "* ]]; then
    echo "E0_GATE=FAIL store=$STORE line=${E0_LINE:-missing}（前缀库未完成，不起跑）"; exit 2
fi
echo "E0_GATE=PASS $E0_LINE"

# ── 评估目录：全新 run 名，不覆盖；DRY_RUN 只写临时目录 ──
if [[ "$DRY" == 1 ]]; then
    PLAN_DIR="$(mktemp -d)"
    echo "DRY_PLAN_DIR=$PLAN_DIR"
else
    [[ ! -e "$EVAL_ROOT" && ! -e "$RETRY_ROOT" ]] || { echo "评估目录已存在，拒绝覆盖：$EVAL_ROOT / $RETRY_ROOT"; exit 2; }
    PLAN_DIR="$EVAL_ROOT/plans"
    mkdir -p "$PLAN_DIR" "$EVAL_ROOT/logs" "$SLOT_DIR"
fi
nm_assert_writable PLAN_DIR "$PLAN_DIR" || exit 2

# ── ② 本次 $SHARDS 片计划（policy 仓 make_shard_plan.py 只读调用，候选库绝对路径）──
make_plan() {  # $1=输出目录 $2=片数
    "${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/make_shard_plan.py" shards "$1" --shards "$2" \
        --candidates "$CANDIDATES" --demo-store "$STORE" 2>&1 | tee "$1/make_shard_plan.log"
    return "${PIPESTATUS[0]}"
}
make_plan "$PLAN_DIR" "$SHARDS"
PLAN_RC=$?
PLAN_LINE="$(grep '^PLAN_OK ' "$PLAN_DIR/make_shard_plan.log")"
[[ "$PLAN_RC" == 0 && -n "$PLAN_LINE" ]] || { echo "PLAN_OK 缺失或 make_shard_plan.py rc=$PLAN_RC，不起跑"; exit 2; }
TOTAL="$(sed -n 's/^PLAN_OK .* total=\([0-9]*\) .*/\1/p' <<< "$PLAN_LINE")"
for ((K=0; K<SHARDS; K++)); do
    echo "SHARD_MAP shard=s$K gpu=$((K % 8)) port_base=$((18011 + K * 7)) plan=$PLAN_DIR/shard$K.json"
done
# ── G11a：片数不是 32 时与 32 片参照计划核对身份集合 ──
if [[ "$MODE" != g6b && "$SHARDS" != 32 ]]; then
    mkdir -p "$PLAN_DIR/ref32"
    make_plan "$PLAN_DIR/ref32" 32 > /dev/null || { echo "32 片参照计划生成失败"; exit 2; }
    py plan_same "$PLAN_DIR/ref32" 32 "$PLAN_DIR" "$SHARDS" | tee "$PLAN_DIR/SHARD_PLAN_SAME.txt"
    [[ "${PIPESTATUS[0]}" == 0 ]] || { echo "SHARD_PLAN_SAME 不过，不起跑"; exit 2; }
fi

if [[ "$MODE" == g6b ]]; then
    # ════ G6b：评估加载冒烟（G10）════
    py g6b_plans "$PLAN_DIR" "$PLAN_DIR" || { echo "G6B 计划生成失败"; exit 2; }
    if [[ "$DRY" == 1 ]]; then
        echo "DRY g6b ① LOCAL_GPU_SHARE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 $SHARD_SH $RUN_NAME $CKPT g6b-single.json $SEED g1"
        echo "DRY g6b ② LOCAL_GPU_SHARE=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 GPU 0：g2a(SHARD_INDEX=0, g6b-a.json) + g2b(SHARD_INDEX=1, g6b-b.json) 同时起"
        echo "DRY_RUN_POOL=PASS mode=g6b writable_in_policy=0"
        exit 0
    fi
    nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$EVAL_ROOT/logs/g6b-mem.csv" 2>&1 &
    MON=$!
    T0=$(date +%s)
    env LOCAL_GPU_SHARE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 \
        bash "$SHARD_SH" "$RUN_NAME" "$CKPT" "$PLAN_DIR/g6b-single.json" "$SEED" g1 \
        > "$EVAL_ROOT/logs/g1.log" 2>&1
    RC1=$?
    T1=$(date +%s)
    echo "SHARD_EXIT_CODE=$RC1 shard=g1 seconds=$((T1 - T0))"
    run_slot 0 g2a 0 "$PLAN_DIR/g6b-a.json" "$RUN_NAME" "$EVAL_ROOT/logs/g2a.log" "$EVAL_ROOT/logs/g2a.rc" &
    P_A=$!
    run_slot 0 g2b 1 "$PLAN_DIR/g6b-b.json" "$RUN_NAME" "$EVAL_ROOT/logs/g2b.log" "$EVAL_ROOT/logs/g2b.rc" &
    P_B=$!
    wait "$P_A"; wait "$P_B"
    T2=$(date +%s)
    kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    RC_A="$(cat "$EVAL_ROOT/logs/g2a.rc")"; RC_B="$(cat "$EVAL_ROOT/logs/g2b.rc")"
    echo "SHARD_EXIT_CODE=$RC_A shard=g2a"; echo "SHARD_EXIT_CODE=$RC_B shard=g2b"
    echo "G6B_TIME single_s=$((T1 - T0)) share2_wall_s=$((T2 - T1))"
    OK1="$(py episode_ok "$EVAL_ROOT/g1" "$POLICY" "$CKPT_ID" "$SEED")"
    OK_A="$(py episode_ok "$EVAL_ROOT/g2a" "$POLICY" "$CKPT_ID" "$SEED")"
    OK_B="$(py episode_ok "$EVAL_ROOT/g2b" "$POLICY" "$CKPT_ID" "$SEED")"
    PEAK="$(py peak "$EVAL_ROOT/logs/g6b-mem.csv")"
    NORM="$(sed -n 's/^NORM_SHA=\([0-9a-f]*\).*/\1/p' "$EVAL_ROOT/g1/server.log" | head -n 1)"
    HS=FAIL; [[ "$RC1" == 0 && "$OK1" == 1 ]] && HS=PASS
    S2=FAIL; [[ "$RC_A" == 0 && "$RC_B" == 0 && "$OK_A" == 1 && "$OK_B" == 1 ]] && S2=PASS
    V=FAIL; [[ "$HS" == PASS && "$S2" == PASS && -n "$NORM" ]] && V=PASS
    grep -h '^SERVE_RNG' "$EVAL_ROOT/g1/server.log" || true
    echo "EVAL_LOAD_SMOKE=$V episodes=1 handshake=$HS share2=$S2 peak_mem_gb=$PEAK norm_sha=${NORM:-missing}" | tee "$EVAL_ROOT/EVAL_LOAD_SMOKE.txt"
    [[ "$V" == PASS ]] || exit 1
    exit 0
fi

if [[ "$MODE" == cap ]]; then
    # ════ 压测：GPU 0 同卡 $PER_GPU 份各 1 局（G11）════
    (( PER_GPU <= SHARDS )) || { echo "EVAL_PER_GPU=$PER_GPU 超过分片数 $SHARDS"; exit 2; }
    py cap_plans "$PLAN_DIR" "$PLAN_DIR" "$PER_GPU" || { echo "压测计划生成失败"; exit 2; }
    if [[ "$DRY" == 1 ]]; then
        for ((K=0; K<PER_GPU; K++)); do
            echo "DRY cap LOCAL_GPU_SHARE=$PER_GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION GPU 0：cap$K(SHARD_INDEX=$K, port_base=$((18011 + K * 7)), cap$K.json) 同时起"
        done
        echo "DRY_RUN_POOL=PASS mode=cap writable_in_policy=0"
        exit 0
    fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 -lms 500 \
        > "$EVAL_ROOT/logs/cap-mem.csv" 2>&1 &
    MON=$!
    T0=$(date +%s)
    PIDS=()
    for ((K=0; K<PER_GPU; K++)); do
        run_slot 0 "cap$K" "$K" "$PLAN_DIR/cap$K.json" "$RUN_NAME" "$EVAL_ROOT/logs/cap$K.log" "$EVAL_ROOT/logs/cap$K.rc" &
        PIDS+=("$!")
    done
    for P in "${PIDS[@]}"; do wait "$P"; done
    T1=$(date +%s)
    kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    FAILED=0
    for ((K=0; K<PER_GPU; K++)); do
        RC="$(cat "$EVAL_ROOT/logs/cap$K.rc")"
        OK="$(py episode_ok "$EVAL_ROOT/cap$K" "$POLICY" "$CKPT_ID" "$SEED")"
        OOM="$(grep -c -E 'out of memory|RESOURCE_EXHAUSTED' "$EVAL_ROOT/logs/cap$K.log" "$EVAL_ROOT/cap$K/server.log" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')"
        echo "SHARD_EXIT_CODE=$RC shard=cap$K episode_ok=$OK oom_lines=$OOM"
        [[ "$RC" == 0 && "$OK" == 1 && "$OOM" == 0 ]] || FAILED=$((FAILED + 1))
    done
    read -r PEAK FREE <<< "$(py cap_mem "$EVAL_ROOT/logs/cap-mem.csv")"
    V=FAIL
    if [[ "$FAILED" == 0 && "$FREE" != nan ]] && awk -v f="$FREE" 'BEGIN {exit !(f >= 4)}'; then V=PASS; fi
    echo "CAP_TIME wall_s=$((T1 - T0)) failed=$FAILED"
    echo "EVAL_CAPACITY=$V per_gpu=$PER_GPU peak_mem_gb=$PEAK free_gb=$FREE" | tee "$EVAL_ROOT/EVAL_CAPACITY.txt"
    [[ "$V" == PASS ]] || exit 1
    exit 0
fi

# ════ 正式评估 ════
if [[ "$DRY" == 1 ]]; then
    py preseed "$PLAN_DIR" "$PRESEED_SRC" - "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" || exit 2
    echo "DRY 正式评估：$SHARDS 片按 SHARD_MAP 同时起（每卡 <= $PER_GPU 片，flock 占位），之后 scan → retry1 → check_shard → merge"
    echo "DRY_RUN_POOL=PASS mode=eval shards=$SHARDS per_gpu=$PER_GPU writable_in_policy=0"
    exit 0
fi
py preseed "$PLAN_DIR" "$PRESEED_SRC" "$EVAL_ROOT" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" \
    | tee "$EVAL_ROOT/logs/preseed.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "PRESEED 不过，不起跑"; exit 2; }

PIDS=()
for ((K=0; K<SHARDS; K++)); do
    run_slot "$((K % 8))" "s$K" "$K" "$PLAN_DIR/shard$K.json" "$RUN_NAME" "$EVAL_ROOT/logs/s$K.log" "$EVAL_ROOT/logs/s$K.rc" &
    PIDS+=("$!")
done
for P in "${PIDS[@]}"; do wait "$P"; done
for ((K=0; K<SHARDS; K++)); do echo "SHARD_EXIT_CODE=$(cat "$EVAL_ROOT/logs/s$K.rc") shard=s$K"; done

py scan "$EVAL_ROOT" "$PLAN_DIR" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" | tee "$EVAL_ROOT/logs/scan.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "分片判定脚本失败"; exit 1; }
if grep -q 'verdict=infra' "$EVAL_ROOT/logs/scan.log"; then
    echo "EVAL692=FAIL reason=infra（见上方 verdict=infra 的分片，留证交用户，不重跑、不合并）"; exit 1
fi
PIDS=()
for ((K=0; K<SHARDS; K++)); do
    [[ -f "$EVAL_ROOT/s$K/retry1-plan.json" ]] || continue
    run_slot "$((K % 8))" "s$K" "$K" "$EVAL_ROOT/s$K/retry1-plan.json" "$RETRY_RUN" \
        "$EVAL_ROOT/logs/s$K-retry1.log" "$EVAL_ROOT/logs/s$K-retry1.rc" &
    PIDS+=("$!")
done
for P in "${PIDS[@]}"; do wait "$P"; done
for ((K=0; K<SHARDS; K++)); do
    [[ -f "$EVAL_ROOT/logs/s$K-retry1.rc" ]] && echo "SHARD_EXIT_CODE=$(cat "$EVAL_ROOT/logs/s$K-retry1.rc") shard=s$K-retry1"
done
py integrate "$EVAL_ROOT" "$RETRY_ROOT" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" | tee "$EVAL_ROOT/logs/integrate.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "EVAL692=FAIL reason=retry_integrate"; exit 1; }
[[ ! -d "$RETRY_ROOT" ]] || rmdir "$RETRY_ROOT" || { echo "重跑根目录未清空：$RETRY_ROOT"; exit 1; }

FAILED=0
for ((K=0; K<SHARDS; K++)); do
    "${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/check_shard.py" "$EVAL_ROOT/s$K" "$POLICY" "$CKPT_ID" "$SEED" "$PLAN_DIR/shard$K.json" \
        > "$EVAL_ROOT/logs/check-s$K.log" 2>&1
    RC=$?
    echo "CHECK_SHARD_EXIT_CODE=$RC shard=s$K $(grep -o '^SHARD_PASS' "$EVAL_ROOT/logs/check-s$K.log")"
    [[ "$RC" == 0 ]] || FAILED=$((FAILED + 1))
done
[[ "$FAILED" == 0 ]] || { echo "EVAL692=FAIL reason=check_shard failed_shards=$FAILED"; exit 1; }
"${CLIENT_UV[@]}" "$POLICY_EVAL_SCRIPTS/merge_shards.py" "$EVAL_ROOT" --policy "$POLICY" --ckpt "$CKPT_ID" --seed "$SEED" \
    --expect-total "$TOTAL" 2>&1 | tee "$EVAL_ROOT/logs/merge.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "EVAL692=FAIL reason=merge"; exit 1; }
py summary "$EVAL_ROOT" | tee "$EVAL_ROOT/EVAL692.txt"
[[ "${PIPESTATUS[0]}" == 0 ]] || exit 1
echo "LOCAL692_DONE $(date -u +%FT%TZ)"
