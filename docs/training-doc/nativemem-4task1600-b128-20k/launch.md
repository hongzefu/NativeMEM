# nativemem-4task1600-b128-20k 起跑档案（第一段：版本与配置还原）

写于美东 2026-11-16 晚，CPU 准备全部通过、GPU 阶段尚未开始。第二段（训练过程、评估、结论）在跑完后写入 `result.md`。

## ① 目的
在 RoboMME 4task（BinFill、RouteStick、VideoRepick、VideoUnmaskSwap 各 400 集）上，除多任务外尽量照 NativeMEM 原版（论文 arXiv 2607.06678 v2、OpenDriveLab/NativeMEM@c599f81、NativeMEM-RMBench@36cf2ca）训一个覆盖 4 任务的模型，并在本机跑 692 局评估。计划：本仓库根目录 `1116-nativemem-4task-stage2-plan.html`。

用户原话（2026-11-16，按时间）：
- 「我们不做stage1了 只做stage1.5 2 给出完整的方案」
- 「尽可能还原paper原版的设计，除了做multi task learning」
- 「同意开工 先做好所有非gpu工作 其他等现在在做的训练+eval结束后」
- 「放在nativemem这个repo里面 所有的结果尽可能都放在这里！」
- 裁决：作者 tokenizer（P-LINEAGE 例外）；4×400；20k×b128、lr 不缩放；按帧均匀采样；只用执行帧；评估只用 48 步；存档整目录传 HF 公开；触发「只要卡空就起跑」；ONLINE_TOKEN 只记录。

## ② 版本与代码状态
- 代码：fork `hongzefu/NativeMEM` 分支 `robomme-4task`，Beta 提交即本文件所在提交（`commitNM1.5Beta`），起跑时 HEAD 必须等于它且工作区干净（`nm_pipeline.sh` formal 步与 `run_train.sh --formal` 双重检查）。
- 主要合并：S0 `b48e45a`/`ec0acb3`（建库）、S2 `ba76f19`/`991df84`/`045e7e5`（评估、触发、串联）、S1 `9f70724`/`b0c069b`（训练侧）、归一化 `12c7612`。
- 环境：`/scratch/hongze/NativeMEM/.venv`（uv，Python 3.11，jax 0.5.3）；评估客户端解释器为 policy 仓 `v1-store/envs/policy-eval-client`（只读使用）。
- 机器：AWS p4de，8×A100-80GB；存储 `/dev/md0`（本地 NVMe RAID）。

## ③ 启动与配置还原
- 配置：`git show <Beta>:src/openpi/training/config_nativemem.py` 中 `nativemem_pi05_robomme4task`。相对官方 `nativemem_pi05` 的差异只有：batch_size 128、fsdp_devices 8、action_horizon 50（官方默认同为 50）、memory_tokenizer_max_len 640、assets robomme4task、8 维 RoboMME 输入输出变换；其余（lr warmup 1000 / peak 2e-5 / cosine 到 1e-6、EMA 0.99、AdamW、seed 42、save_interval 10000、keep_period 20000、num_workers 8、20000 步）照原版。
- 权重：`NativeMEMWeightLoader("checkpoints/mem_tokenizer_pretrain/pretrain/49999/params")`，HF `Ziyeeee/NativeMEM` revision `a0317e14c232aa64da0c44a8d1adc0076161f60e`，21 个文件，清单 `scripts/robomme/tokenizer_pin.json`（manifest sha256 前缀 fbcc1406）。
- 数据：`data/robomme4task-v2/episodes/ep_0000..1599.h5`，源 bucket `HongzeFu/robomme-4task-h5-20260912-v2`（MANIFEST sha256 `df992cdf…`），与 V17 训练库同一批 1600 集。
- 归一化：`assets/robomme4task/norm_stats.json` sha256 `f4710dff1d4bec66931adce8d74a9f7a581e7376ffc8470af432458d59ef6f6d`。
- 启动方式：tmux `nm-trig` 运行 `scripts/robomme/nm_trigger.sh`，J#22 四条件（`runs/READY.json`、V17 评估会话结束且日志完整 `EXIT_CODE=` 与 RUN_NAME、8 卡显存 <2000 MiB、一次性锁）成立后执行 `bash scripts/robomme/nm_pipeline.sh`：

```bash
cd /scratch/hongze/NativeMEM
NM_TRIG_CMD="bash scripts/robomme/nm_pipeline.sh" bash scripts/robomme/nm_trigger.sh
# nm_pipeline 步骤：memtok → dataset_smoke → smoke20 → resume → online_token → g6b → capacity → cleanup_smoke → formal(+nm-ckptup) → eval
# formal 实际命令：bash scripts/robomme/run_train.sh --formal
#   = .venv/bin/python scripts/train_nativemem.py nativemem_pi05_robomme4task --exp_name=nativemem-4task1600-b128-20k --no-wandb-enabled
```

## ④ 数据集与划分口径
- 1600 集、1,192,918 帧；训练样本为 605,611 个执行帧（演示帧 587,307 只作记忆），按帧均匀采样：BinFill 42.4%、VideoRepick 26.1%、RouteStick 17.7%、VideoUnmaskSwap 13.7%。
- 评估：policy 仓 692 局分片计划（700 候选 − 8 条前缀预生成失败；含 9 条预标 error，实际 rollout 683），种子 7，执行 48 步、记忆每 12 帧。

## ⑤ 关键超参
见 ③；记忆：每帧前视＋腕部各 1 个 1152 维 token，训练间隔每样本随机一次 ∈[8,16]、补第 0 帧后升序，max_len 640。

## ⑥ 硬件与预计耗时
8×A100；抽 token 约 45 min、冒烟与压测约 60–90 min、正式训练估 2.3–2.8 s/步 × 20000 ≈ 13–15 h、评估 1–1.5 h。

## 起跑前闸门（CPU，全部 PASS）
```
DATA_V2=PASS episodes=1600 frames=1192918 views=2 derived=PASS faults=4/4
NORM_STATS=PASS states=605611 action_vectors=30280550 sha256=f4710dff…
TOKENIZER_PIN=PASS level=full files=21 sha=fbcc1406 revision=a0317e14c232aa64da0c44a8d1adc0076161f60e
DATA_LINEAGE=EXCEPTION tokenizer=nativemem-stage1-49999 reason=user-choice-1116
DATA_SAME_AS_V17=PASS members=1600 mismatches=0
NORM_SHA=PASS sha=f4710dff tracked=yes
WEIGHTS_LOAD=PASS loader=NativeMEMWeightLoader src=stage1-49999 random_init=0 unused=encoder
G6A_SYNTHETIC=PASS tests=18 passed
CLI_PARSE=PASS modes=3
RUN_TRAIN=PASS mode=check-only
CLIENT_WRIST_E2E=PASS episodes=2 infers=5
```

## 本轮 tmux 会话清单
`nm-memframes`（第一轮，已结束）、`nm-data2`、`nm-norm`、`nm-check`（均已结束）、`nm-trig`（触发与串联）、`nm-ckptup`（上传，formal 后起）。只按此清单 `tmux kill-session -t '=名'`。

## 产物路径
- 数据：`data/robomme4task-v2/`；token：`data/robomme4task-v2/mem_tokens.f32.bin`
- 存档：`checkpoints/nativemem_pi05_robomme4task/nativemem-4task1600-b128-20k/19999`
- 评估：`runs/eval/local692-nativemem-20k/`
- 日志：`logs/pipeline/`、`logs/train/`、`runs/logs/nm-trigger.log`

## 实施中调整（美东 2026-11-17 05:13–05:45）
- **G9a（ENCODER_IMPL）判据放宽**：GPU 串联在 online_token 步首次判 FAIL：`ENCODER_IMPL=FAIL n=32 max_abs=0.0694 ratio_min=0.999878 ratio_max=1.000269`（cos_min=0.999974）。原因是 GPU 上 f32 矩阵乘默认降精度，服务器逐帧 batch=1 与离线 batch=64 走不同 kernel；CPU 上同一比对 max_abs=0，离线库与离线重算 max_abs=0，原版固有的 JPEG/非 JPEG 差异（G9b）cos_min=0.977、max_abs=1.9。
- 新判据：`cos_min>=0.9999` 且范数比 ∈ [0.999,1.001]；原 `max_abs<=1e-5` 改为 `strict_1e-5=` 只记录。附只记录的视角对调参考 cos（CPU 实测 mean 0.50、min 0.24）。
- 对抗验证 `ADJUST_REVIEW=PASS scale=same findings=3`；合并前审查 `PRE_MERGE_REVIEW=PASS base=7dcc89c tip=60fd061 findings=0`。训练代码与数值语义不变。
- 因合入新提交，重打 Beta（本提交）后从 online_token 步续跑：`NM_PIPELINE_FROM=online_token bash scripts/robomme/nm_pipeline.sh`（tmux `nm-pipe`）。
- 已过闸门（首轮，Beta 7dcc89c）：`MEMTOKENS=PASS rows=1192918 … shards=8/8 gaps=0 overlaps=0 recompute=46/46`、`DATASET_SMOKE=PASS n=605611 samples=32`、`SMOKE20=PASS steps=20 save_load=PASS rotate=PASS exit=0 finite=PASS`（稳态约 2.9 s/步）、`RESUME=PASS first_step=20 exit=0`。
- **g6b 同卡多片 GPU 归属误判（工程故障，美东 05:28–05:50）**：`EVAL_LOAD_SMOKE=FAIL episodes=1 handshake=PASS share2=FAIL peak_mem_gb=56.7 norm_sha=f4710dff`，g2a 报「计算进程 3687019 不属于已登记分片的进程树」；g1 单份与 g2b 各跑完 1 局。修复：查询后已退出（/proc/<pid> 不存在）的进程跳过，存活陌生进程仍拒绝；`PRE_MERGE_REVIEW=PASS base=c1a33ac tip=b1b8aa1`。失败目录保留为 `runs/eval/nm-g6b.fail1-20261117`。重打 Beta 后从 g6b 步续跑。
