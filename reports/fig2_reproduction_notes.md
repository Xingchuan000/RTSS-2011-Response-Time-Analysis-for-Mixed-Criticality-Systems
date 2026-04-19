# FIG2 Reproduction Notes

- 生成时间: 2026-04-16T15:45:12
- figure: fig2
- mode: paper
- num_tasksets: 200
- seed: 2026
- num_tasks(default): 20
- num_tasks(runtime): 20
- total_util(default): 0.8
- period_range: [10, 1000]
- cf(default): 2.0
- cp(default): 0.5
- time_scale: 100
- x_axis: cf
- y_axis_metric: weighted_schedulability
- deadline_mode(default): implicit
- deadline_mode(runtime): implicit
- deadline_ratio_min(default): 0.5
- criticality_assignment(default): bernoulli
- lo_hi_budget_policy(default): scaled_by_cf

## 输出文件
- outputs/fig2/raw_results.csv
- outputs/fig2/aggregated_results.csv
- outputs/fig2/fig2.png