# assets/robomme4task（占位）

本目录是配置 `nativemem_pi05_robomme4task` 的归一化统计落点（`AssetsConfig(assets_dir="./assets", asset_id="robomme4task")`）。

尚未生成。由主会话在主检出、每集 H5 重抽完成（G1）且 S1 合入后用 CPU 生成（G4）：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/robomme/compute_norm_stats_robomme.py
# 末行：NORM_STATS=PASS states=605611 action_vectors=30280550 sha256=<…>
```

产出两份文件，均须 `git add -f`（本目录在 `.gitignore` 的 `assets/*` 内）并在 Beta 提交前入库：

- `norm_stats.json`：state 与 50 步 delta 动作（前 7 维减当前 state，夹爪绝对）的 mean／std／q01／q99，8 维；
- `norm_stats.provenance.json`：计数、norm_stats.json 的 sha256、数据来源与 episode_manifest 的 sha256。

`scripts/robomme/run_train.sh` 起跑前核对两份文件已入 git、与 HEAD 相同、计数为 605611／30280550（G4 NORM_STATS／NORM_SHA）。
