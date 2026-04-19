# 论文复现实验入口说明（阶段4）

## 1. 统一入口

项目提供统一脚本：

- `scripts/reproduce_rtss11.py --figure fig1|fig2|fig3|fig4|fig5`

常用参数：

- `--mode fast|paper`：实验规模模式
- `--num-tasksets`：每个 sweep 点任务集数量
- `--seed`：基础随机种子
- `--output-root`：输出根目录（默认 `outputs`）
- `--config`：可选自定义生成器 YAML

## 2. 单图运行命令

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig1 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig2 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig3 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig4 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig5 --mode fast --num-tasksets 80 --seed 2026
```

## 3. 输出目录约定

每张图输出到：`outputs/figX/`

至少包含：

- `raw_results.csv`
- `aggregated_results.csv`
- `figX.png`

其中 Fig.2~Fig.4 额外输出：

- `util_layer_aggregated.csv`

## 4. 方法与优先级策略

默认比较的方法集合：

- `UB-H&L`（`ub_hl + dm`）
- `AMC-max`（`amc_max + opa`）
- `AMC-rtb`（`amc_rtb + opa`）
- `SMC`（`smc + opa`）
- `SMC-NO`（`smc_no + opa`）
- `CrMPO`（`crmpo_baseline + crmpo`）

## 5. 复现说明文档

每次运行后会自动更新对应说明：

- `reports/fig1_reproduction_notes.md`
- `reports/fig2_reproduction_notes.md`
- `reports/fig3_reproduction_notes.md`
- `reports/fig4_reproduction_notes.md`
- `reports/fig5_reproduction_notes.md`
