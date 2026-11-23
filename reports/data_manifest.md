# 数据清单 — playground-series-s6e11 (Predicting Credit Card Fraud)

- 拉取时间：2026-11-07
- 来源：`kaggle competitions download -c playground-series-s6e11`
- 本地路径：`data/playground-series-s6e11/`

## 文件

| 文件 | 行数（含表头） | 数据行数 | 列数 | 大小 |
|---|---|---|---|---|
| `train.csv` | 699,961 | 699,960 | 26 | 98,533,176 B |
| `test.csv` | 299,985 | 299,984 | 25 | 41,068,069 B |
| `sample_submission.csv` | 299,985 | 299,984 | 2 | 8,399,564 B |

列名与官方 `competitions files` 输出一致，未缺失。

## 结构

- 主键 `id`：train 为 `0 … 699,959` 连续；test 为 `699,960 … 999,943` 连续。两者无交集。
- 目标 `is_fraud`：train 独有。
- test 比 train 少 `is_fraud`，其余 24 个特征列完全一致。

## 提交格式

`sample_submission.csv` 列为 `id, is_fraud`（预测概率）。指标 ROC AUC。

## 注意

- 数据中不存在缺失值、完全重复行、train/test 特征重叠。
- 原始文件只读，任何清洗产物写 `data/playground-series-s6e11/processed/`。
