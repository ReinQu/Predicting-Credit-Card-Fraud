"""探测数据生成结构。

问题 1：数值列之间是否存在确定性关系？（用其余列预测某一列，看 R²）
        R² 接近 1  -> 该列是派生出来的，数据按公式生成
        R² 接近 0  -> 该列独立随机

问题 2：离散特征的组合空间有多大？同一组合下的标签是否一致？
        若大量组合内标签 100% 一致  -> 标签是确定函数，模型还远未到天花板
        若组内标签混杂            -> 标签含真实随机性，0.947 已接近上限

用法: python -I src/probe_generation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"

TARGET = "is_fraud"
ID = "id"

TARGETS_TO_PROBE = [
    "merchant_risk_score",
    "velocity_score",
    "distance_from_home_km",
    "hours_since_last_txn",
    "amount_usd",
    "account_balance_usd",
]

DISCRETE = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "cvv_retry_count", "prior_disputes", "txn_count_last_24h",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def probe_derivation(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("问题 1：用其余 23 列预测每一个数值列，看 R²")
    print("=" * 78)
    print(f"{'目标列':<26}{'R²':>10}{'RMSE':>14}{'基线RMSE':>14}{'判定':>12}")
    print("-" * 78)

    for tgt in TARGETS_TO_PROBE:
        feats = [c for c in df.columns if c not in (ID, TARGET, tgt)]
        X = df[feats].copy()
        y = df[tgt].to_numpy()

        # 离散列转 category，其余保持数值
        for c in feats:
            if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
                X[c] = X[c].astype("category")
            elif X[c].dtype == bool:
                X[c] = X[c].astype("int8")

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
        m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=64,
            min_child_samples=100, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, n_jobs=-1, verbose=-1, random_state=42,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict(Xva)
        r2 = r2_score(yva, pred)
        rmse = float(np.sqrt(((pred - yva) ** 2).mean()))
        base = float(np.sqrt(((yva - ytr.mean()) ** 2).mean()))
        verdict = "派生" if r2 > 0.9 else ("部分相关" if r2 > 0.3 else "独立")
        print(f"{tgt:<26}{r2:>10.4f}{rmse:>14.3f}{base:>14.3f}{verdict:>12}")


def probe_discrete(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("问题 2：离散特征组合的重复度与标签纯度")
    print("=" * 78)

    key = df[DISCRETE].astype(str).agg("|".join, axis=1)
    grp = df.groupby(key, observed=True)[TARGET].agg(["size", "mean"])

    print(f"离散列数: {len(DISCRETE)}")
    print(f"行数: {len(df):,}")
    print(f"不同组合数: {len(grp):,}   (平均每组 {len(df) / len(grp):.2f} 行)")
    print(f"出现 1 次的组合: {(grp['size'] == 1).sum():,} ({(grp['size'] == 1).mean():.1%})")
    print(f"出现 >=10 次的组合: {(grp['size'] >= 10).sum():,}")

    big = grp[grp["size"] >= 10]
    print(f"\n在 >=10 行的 {len(big):,} 个组合中：")
    print(f"  标签全为 0 的组合: {(big['mean'] == 0).sum():,} ({(big['mean'] == 0).mean():.1%})")
    print(f"  标签全为 1 的组合: {(big['mean'] == 1).sum():,} ({(big['mean'] == 1).mean():.1%})")
    print(f"  标签混杂的组合:    {((big['mean'] > 0) & (big['mean'] < 1)).sum():,}")
    print(f"  覆盖行数: {int(big['size'].sum()):,} ({big['size'].sum() / len(df):.1%})")

    # 若标签是确定函数，上面的"全0/全1"比例会非常高
    pure = ((big["mean"] == 0) | (big["mean"] == 1)).mean()
    print(f"\n判定: 纯组合占比 {pure:.1%}", end="  ->  ")
    if pure > 0.9:
        print("标签接近确定函数，天花板远高于当前")
    elif pure > 0.5:
        print("标签有强结构但含噪声")
    else:
        print("标签在同一离散组合内高度混杂，随机性大")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    df = pd.read_csv(DATA / "train.csv")
    print(f"读取 {df.shape}\n")

    probe_derivation(df)
    probe_discrete(df)
    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
