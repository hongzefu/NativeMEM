# nativemem-4task1600-b128-20k 结果档案（第二段：训练过程、评估、结论）

写于美东 2026-11-17 晚，第一段（版本与配置还原、起跑前闸门、实施中调整）见同目录 `launch.md`。

## ① 一句话结论与指标速览
NativeMEM（作者 Stage1 tokenizer + 本仓库 Stage2，4 任务一个模型，20k 步 × batch 128）在本机 692 局评估成功 **290/692 = 41.91%**（macro 0.4180），比同一 692 局计划、同一种子 7 的 V17（context motion160 50k）**254/692 = 36.71%** 高 5.2 个百分点；但本次每次推理执行 48 步、V17 执行 16 步，两者不是同条件对照（计划 X11），差异不能单独归因于记忆结构。

```
TRAIN_FINAL=PASS step=19999 ckpts=1 finite=PASS exit=0 ckpt_dirs=[19999 ]
HF_VERIFY=PASS ckpts=1 files=78 bytes=42486972074
EVAL692=PASS episodes=692 success=290 rate=0.4191 preseeded_errors=9 new_errors=0 retried=0
NM_PIPELINE=PASS 2026-11-17 23:09:43 美东
```

## ⑥ 硬件与耗时（实测）
| 阶段 | 起止（美东 2026-11-17） | 耗时 |
|---|---|---|
| 正式训练 20000 步 | 05:48 → 22:26 | 16 h 36 min（tqdm elapsed 16:36:13），稳态 2.8–3.2 s/步 |
| 19999 存档 | 22:25:51 → 22:26:14 | 22.8 s（阻塞 15.8 s） |
| HF 上传 + 读回（tmux `nm-ckptup`） | 22:28 → 22:32 | 约 4 min，78 文件 42.49 GB |
| 692 局评估（32 片、每卡 4 份） | 22:28 → 23:09 | 41.6 min |

计划估训练 13–15 h（2.3–2.8 s/步），实测慢约 15%；评估估 1–1.5 h，实测 42 min（执行 48 步、推理次数约为 16 步的 1/3）。

## ⑦ 训练过程行为
逐千步（`records/train.summary.log`，共 200 条 Step 行；全程无 nan/NaN、无 Traceback、无 OOM，退出码 0）：

| 步 | action_loss | grad_norm | 步 | action_loss | grad_norm |
|---|---|---|---|---|---|
| 0 | 0.134 | — | 10000 | 0.0021 | 0.0278 |
| 1000 | 0.0085 | 0.0616 | 11000 | 0.0018 | 0.0258 |
| 2000 | 0.0059 | 0.0419 | 12000 | 0.0017 | 0.0286 |
| 3000 | 0.0051 | 0.0418 | 13000 | 0.0014 | 0.0243 |
| 4000 | 0.0042 | 0.0343 | 14000 | 0.0013 | 0.0251 |
| 5000 | 0.0036 | 0.0356 | 15000 | 0.0011 | 0.0225 |
| 6000 | 0.0032 | 0.0308 | 16000 | 0.0011 | 0.0231 |
| 7000 | 0.0029 | 0.0328 | 17000 | 0.0010 | 0.0211 |
| 8000 | 0.0026 | 0.0335 | 18000 | 0.0009 | 0.0224 |
| 9000 | 0.0023 | 0.0282 | 19000 | 0.0008 | 0.0203 |

loss 单调下降到末段仍在降（cosine 尾部 lr→1e-6），param_norm 1804.39→1804.47 基本不变（全参微调、lr 峰值 2e-5）。10000 步存档按 keep_period 20000 在 19999 保存后被轮转删除，只留 19999（与计划 G12 `ckpts=1` 一致）。

## ⑧ 训练后评估
口径：policy 仓 692 局分片计划（700 候选 − BinFill 8 条前缀预生成失败；9 条预标 error 与 V11/V16/V17 同一组，实际 rollout 683）、种子 7、max_steps 2000、执行 48 步、记忆每 12 帧；服务器每局 reset 重播 rng=key(7)。`PLAN_OK shards=32 total=692 per_shard=21 disjoint=True`；32 片 `SHARD_EXIT_CODE=0` 与 `CHECK_SHARD_EXIT_CODE=0 … SHARD_PASS` 全齐；`RETRY_INTEGRATE=PASS retried=0`。

| 组 | NativeMEM 20k（48 步） | V17 ctx-motion160 50k（16 步） |
|---|---|---|
| BinFill/easy | 43/49 = 0.878 | 40/49 = 0.816 |
| BinFill/medium | 25/49 = 0.510 | 21/49 = 0.429 |
| BinFill/hard | 9/44 = 0.205 | 4/44 = 0.091 |
| RouteStick/easy | 42/50 = 0.840 | 47/50 = 0.940 |
| RouteStick/medium | 43/50 = 0.860 | 38/50 = 0.760 |
| RouteStick/hard | 32/50 = 0.640 | 24/50 = 0.480 |
| RouteStick/xhard | 8/50 = 0.160 | 3/50 = 0.060 |
| VideoRepick/easy | 19/50 = 0.380（err 1） | 13/50 = 0.260（err 1） |
| VideoRepick/medium | 14/50 = 0.280（err 3） | 11/50 = 0.220（err 3） |
| VideoRepick/xhard | 17/50 = 0.340（err 3） | 11/50 = 0.220（err 3） |
| VideoUnmaskSwap/easy | 11/50 = 0.220 | 13/50 = 0.260 |
| VideoUnmaskSwap/medium | 15/50 = 0.300 | 16/50 = 0.320 |
| VideoUnmaskSwap/hard | 3/50 = 0.060 | 7/50 = 0.140 |
| VideoUnmaskSwap/xhard | 9/50 = 0.180（err 2） | 6/50 = 0.120（err 2） |
| **合计** | **290/692 = 41.91%** | **254/692 = 36.71%** |

按任务：BinFill 77/142（54.2%，V17 45.8%）、RouteStick 125/200（62.5%，V17 56.0%）、VideoRepick 50/150（33.3%，V17 23.3%）、VideoUnmaskSwap 38/200（19.0%，V17 21.0%）。提升集中在 BinFill、RouteStick 难档与 VideoRepick；VideoUnmaskSwap 略低于 V17。V17 数字出自 policy 仓 `docs/training-doc/local692-ctx-motion160-50k/records/v17-eval.summary.log`（commit `8181918`）。

## ⑨ 用户决策记录（本段新增，第一段已列的不重复）
- 「同意开工 先做好所有非gpu工作 其他等现在在做的训练+eval结束后 你发个message给训练+eval的对话 做好交接工作」（2026-11-16，开工令）。
- 「放在nativemem这个repo里面 所有的结果尽可能都放在这里！」——本档案与 records 均在 fork `robomme-4task` 分支。

## ⑩ 计划外事件与处置
实施中三项调整（G9a 判据放宽、g6b 同卡 GPU 归属误判修复、两次重打 Beta 续跑）已写在 `launch.md`「实施中调整」，正式训练之后无新事件：formal、上传、评估一次通过，无重跑。汇报时刻曾按 EDT（UTC−4）换算多报 1 小时，11 月应为 EST（UTC−5），本档案时刻均已按 EST 写。

## ⑪ 结论与下一步
- 结论：除多任务外照原版配方、只训 20k×128 的 NativeMEM，在本机 692 局上达到 41.91%，高于此前最好的 V17 36.71%；但执行步数 48 vs 16 不同，记忆结构的贡献未被隔离。
- 可选下一步（未授权、待用户决定）：同一 19999 存档按 16 步执行另做一次 692 局对照（需另选与 16 兼容的记忆间隔，计划 X11）；或 V17 按 48 步执行对照。

## ⑫ 归档文件清单（`records/`）
| 文件 | 内容 |
|---|---|
| `train.summary.log` | 训练日志清洗版（748 KB/5488 行 → 80 KB/483 行，200 条 Step 行零丢失） |
| `nm-pipe2.summary.log` | 串联全程判定行（memtok→eval） |
| `nm-ckptup.summary.log` | 上传与读回校验 |
| `eval.summary.log` | 692 局评估 runner 日志（含分组表） |
| `EVAL692.txt`、`merged-{progress,log,shards,errors,retry_plan}.json`、`preseeded-errors.json`、`retries.json` | 评估合并结果 |

产物（不进 git）：存档 `checkpoints/nativemem_pi05_robomme4task/nativemem-4task1600-b128-20k/19999`（远端 HF bucket `HongzeFu/robomme-vla-4task-nativemem-20k-v1`，`HF_VERIFY=PASS`）；评估目录 `runs/eval/local692-nativemem-20k/`（含逐局视频）。
