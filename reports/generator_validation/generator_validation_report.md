# Generator Validation Report

## 1. 配置概览

- num_tasksets: 200
- num_tasks: 20
- total_util: 0.8
- period_range: [10, 1000]
- time_scale: 100
- cf: 2.0
- cp: 0.5
- deadline_mode: implicit
- deadline_ratio_min: 0.5
- criticality_assignment: bernoulli
- lo_hi_budget_policy: scaled_by_cf

## 2. 分布摘要

- HI ratio: {'min': 0.25, 'p25': 0.45, 'median': 0.5, 'p75': 0.6, 'max': 0.8, 'mean': 0.50325, 'std': 0.1063105709701533}
- period: {'min': 1000.0, 'p25': 3000.0, 'median': 9400.0, 'p75': 30600.0, 'max': 100000.0, 'mean': 20990.975, 'std': 24963.11771492846}
- deadline_ratio: {'min': 1.0, 'p25': 1.0, 'median': 1.0, 'p75': 1.0, 'max': 1.0, 'mean': 1.0, 'std': 0.0}
- actual_util_lo: {'min': 0.7987778825960871, 'p25': 0.799764444606468, 'median': 0.8000467088773334, 'p75': 0.8003225680444227, 'max': 0.8010816315436122, 'mean': 0.8000239701363847, 'std': 0.0004319120315227198}
- actual_util_hi: {'min': 0.1626299264350886, 'p25': 0.6351553150926178, 'median': 0.8185613555557871, 'p75': 0.9856638839507691, 'max': 1.4374859287833257, 'mean': 0.8157333259087001, 'std': 0.24277682567648576}

## 3. 输出文件

- taskset_stats.csv: reports/generator_validation/taskset_stats.csv
- task_stats.csv: reports/generator_validation/task_stats.csv
- plot: reports/generator_validation/generator_validation_plots.png