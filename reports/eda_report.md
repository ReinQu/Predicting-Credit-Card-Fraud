# EDA 报告 — Predicting Credit Card Fraud (playground-series-s6e11)

生成脚本：`src/eda_profile.py`　原始输出：`reports/eda_profile.txt`、`reports/eda_profile.json`

## 1. 数据规模

| | 行数 | 列数 |
|---|---|---|
| train | 699,960 | 26（含 `id`、`is_fraud`）|
| test | 299,984 | 25（含 `id`）|

- **无缺失值**：所有列 train/test 缺失率均为 0。
- **无重复**：特征完全重复行 0 条；train/test 特征重叠 0 行。
- **无漂移**：逐特征比较 train/test，最大标准化均值差 SMD = 0.0052（`day_of_week`）；类别特征最大 TV 距离 0.0023。可以认为同分布。

## 2. 目标

- `is_fraud` 正类率 **1.670%**（11,689 / 699,960），不平衡比约 **1 : 60**。
- 指标为 **ROC AUC**，是排序指标，因此正负样本比例不影响绝对值，不需要重采样来"修正"分布。

## 3. 关键发现

### 3.1 ID 只是行号，没有时间结构

train 的 `id` 是 `0…699,959` 连续块，test 是 `699,960…999,943` 连续块。但按 ID 十等分后正类率在 1.59%–1.73% 之间波动，**没有时间漂移**。ID 与目标相关性 -0.0006。

→ 这说明切分不是"按时间前 70% 训练、后 30% 测试"，而是近似随机切分。**可以用 StratifiedKFold**，不必做时间序列验证。

### 3.2 单特征区分力

| 特征 | 单特征 AUC | 说明 |
|---|---|---|
| `merchant_risk_score` | 0.749 | 商户风险分，最强单特征 |
| `velocity_score` | 0.711 | 交易速度分 |
| `cvv_retry_count` | 0.708 | 0→0.93%，1→8.75%，2→41.7% |
| `is_new_merchant` | 0.642 | |
| `customer_age` | 0.625 | |
| `txn_count_last_24h` | 0.620 | 单调：0 次 0.23% → 7 次 4.4% |
| `ip_country_mismatch` | 0.573 | True 16.5% vs False 1.4% |
| `hours_since_last_txn` | 0.568 | |
| `merchant_category` | 0.567 | Gift Cards 7.3%、Crypto 6.5% vs Groceries 1.1% |
| —— | —— | —— |
| `card_type` / `device_type` / `channel` | 0.502–0.506 | **几乎无信号** |
| `distance_from_home_km` / `card_age_months` / `time_of_day_hour` / `day_of_week` | 0.500–0.504 | **几乎无信号** |
| `account_balance_usd` | 0.506 | 无信号（且偏度 14.4）|

### 3.3 布尔风险标志的联合效应

| 标志 | 为 True 时正类率 | 为 False 时 | True 占比 |
|---|---|---|---|
| `ip_country_mismatch` | 16.5% | 1.4% | 1.6% |
| `is_ai_generated_scam_attempt` | 15.2% | 1.6% | 0.33% |
| `billing_shipping_mismatch` | 13.0% | 1.5% | 1.3% |
| `is_foreign_transaction` | 12.6% | 1.5% | 1.7% |
| `used_vpn` | 6.6% | 1.5% | 3.4% |
| `is_new_merchant` | 4.7% | 1.1% | 15.2% |

这些是稀疏但高纯度的信号，GBDT 天然擅长，不必手工做组合。

### 3.4 类别特征信号集中在两列

- `auth_method`：`No Authentication` 6.4% vs `Biometric` 0.5% —— 强
- `merchant_category`：Gift Cards / Crypto / Gaming > 5% vs Groceries 1.1% —— 强
- `channel`、`device_type`、`card_type` 基本无信号

### 3.5 数值特征注意点

- `amount_usd`（偏度 7.0）、`account_balance_usd`（偏度 14.4）长尾明显。**树模型不受影响**，只有在做线性模型/神经网络时才需要 log1p。
- 数值特征之间无 |r|>0.8 的强共线。
- 多个"数值"列其实是低基数离散值（`cvv_retry_count` 4 值、`prior_disputes` 5 值、`txn_count_last_24h` 12 值），可以当数值用，也可以当类别用——留给 baseline 对比。

## 4. 结论与下一步

1. **验证方案**：`StratifiedKFold`，正类率约 1.67%，5 折每折约 2,338 个正例，足够稳定。不需要时间序列切分。
2. **基线**：LightGBM + 原始特征 + 类别特征转 category dtype，不重采样，直接优化 AUC。预期 CV AUC ≈ 0.93–0.95。
3. **特征工程方向**（按性价比排序）：
   - 比值类：`amount_usd / account_balance_usd`、`amount_usd / (velocity_score+1)`
   - 计数类标志求和（6 个布尔风险标志的 sum）、`cvv_retry_count × is_new_merchant` 之类
   - 分箱统计：`merchant_risk_score`、`velocity_score` 的分位数分箱
4. **可以暂缓的列**：`card_type`、`device_type`、`channel`、`distance_from_home_km`、`card_age_months`、`time_of_day_hour`、`day_of_week`、`account_balance_usd`——先跑一次特征消融确认，不要凭单特征 AUC 直接删（可能有交互）。
5. **风险**：本赛无奖牌（Playground 系列不计 progression），`reports/` 里的目标是"把流程走完整 + 产出可复用 pipeline"，不是冲榜。
