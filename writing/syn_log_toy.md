# `log/syn/toy.txt` — 最终测试结果

| 数据文件 | 模型 | Test MSE | Test MAE |
|----------|------|----------|----------|
| trend_150.npy | PT_syn_trend | 0.5169 | 0.5082 |
| trend_150.npy | PT_forecast_v15 | 0.5616 | 0.5324 |
| trend_150.npy | DLinear | 0.9192 | 0.7629 |
| trend_150.npy | BVAR | 1.9085 | 1.1671 |
| periodicity_150.npy | PT_syn_period | 0.2284 | 0.2826 |
| periodicity_150.npy | PT_forecast_v15 | 0.2356 | 0.2884 |
| periodicity_150.npy | DLinear | 0.4546 | 0.5456 |
| periodicity_150.npy | BVAR | 1.0762 | 0.8639 |
| lag_8_150.npy | PT_syn_lag | 0.2190 | 0.2544 |
| lag_8_150.npy | PT_forecast_v15 | 0.2616 | 0.3139 |
| lag_8_150.npy | DLinear | 0.4650 | 0.4763 |
| lag_8_150.npy | BVAR | 0.7762 | 0.6638 |
