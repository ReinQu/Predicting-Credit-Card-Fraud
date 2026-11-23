"""特征工程分批测试：每批单独加入，与当前基线（B_plus 参数）对比。

基线参数 = B_plus（verify_params 中两 seed 均值 0.946870）。
所有实验共用同一套 5 折（seed 42），单次运行约 40s。

用法: python -I src/feature_batches.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_lgbm import ID, N_SPLITS, PARAMS, SEED, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"

B_PLUS = {
    **PARAMS,
    "num_leaves": 24,
    "min_child_samples": 500,
    "reg_lambda": 10.0,
    "colsample_bytree": 0.5,
}

RISK_FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

ORIG_COLS: list[str] = []

BATCHES: dict[str, list[str]] = {
    "B1_ratio": ["amt_bal_ratio", "is_overdraft", "amt_risk", "amt_velocity", "balance_after"],
    "B2_flags": ["risk_flag_sum", "mismatch_sum", "cvv_x_flags", "risk_x_velocity"],
    "B3_interact": ["dist_x_foreign", "dist_x_vpn", "txn24_x_amt"],
    "B4_catcombo": ["cat_x_auth"],
}


def add_features(X: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """就地添加所有候选特征，返回新增列名。"""
    for df in (X, X_test):
        df["amt_bal_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
        df["is_overdraft"] = (df["amount_usd"] > df["account_balance_usd"]).astype("int8")
        df["amt_risk"] = df["amount_usd"] * df["merchant_risk_score"] / 100
        df["amt_velocity"] = df["amount_usd"] / (df["velocity_score"] + 1)
        df["balance_after"] = df["account_balance_usd"] - df["amount_usd"]

        df["risk_flag_sum"] = df[RISK_FLAGS].sum(axis=1)
        df["mismatch_sum"] = df["ip_country_mismatch"] + df["billing_shipping_mismatch"]
        df["cvv_x_flags"] = df["cvv_retry_count"] * df["risk_flag_sum"]
        df["risk_x_velocity"] = df["merchant_risk_score"] * df["velocity_score"] / 100

        df["dist_x_foreign"] = df["distance_from_home_km"] * df["is_foreign_transaction"]
        df["dist_x_vpn"] = df["distance_from_home_km"] * df["used_vpn"]
        df["txn24_x_amt"] = df["txn_count_last_24h"] * df["amount_usd"]

        df["cat_x_auth"] = (
            df["merchant_category"].astype(str) + "_" + df["auth_method"].astype(str)
        )

    cats = sorted(set(X["cat_x_auth"]) | set(X_test["cat_x_auth"]))
    X["cat_x_auth"] = pd.Categorical(X["cat_x_auth"], categories=cats)
    X_test["cat_x_auth"] = pd.Categorical(X_test["cat_x_auth"], categories=cats)

    new = [c for c in X.columns if c not in ORIG_COLS]
    nunique = {c: int(X[c].nunique()) for c in new}
    print(f"新增 {len(new)} 列: {new}")
    print(f"基数: {nunique}")
    print(f"唯一类别组合数: {len(cats)}")
    return new


def run(cols: list[str], X, y, folds, X_test) -> tuple[float, list[float], list[int]]:
    oof = np.zeros(len(X))
    fold_auc, iters = [], []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**B_PLUS)
        m.fit(
            X[cols].iloc[tr],
            y[tr],
            eval_set=[(X[cols].iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict_proba(X[cols].iloc[va])[:, 1]
        fold_auc.append(float(roc_auc_score(y[va], oof[va])))
        iters.append(int(m.best_iteration_ or B_PLUS["n_estimators"]))
    return float(roc_auc_score(y, oof)), fold_auc, iters


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    global ORIG_COLS
    X, y, X_test, _ = prepare()
    ORIG_COLS = list(X.columns)

    add_features(X, X_test)
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

    base_auc, base_folds, base_iters = run(ORIG_COLS, X, y, folds, X_test)
    print(f"\n{'baseline(B_plus, 原始24列)':<28} OOF {base_auc:.6f}  iters {base_iters}")

    results = [{"name": "baseline_orig", "n_feat": len(ORIG_COLS), "oof_auc": base_auc,
                "fold_auc": base_folds, "best_iters": base_iters}]

    for name, cols in BATCHES.items():
        t0 = time.time()
        auc, fauc, iters = run(ORIG_COLS + cols, X, y, folds, X_test)
        print(f"{name:<28} OOF {auc:.6f}  Δ {auc - base_auc:+.6f}  iters {iters}  ({time.time() - t0:.0f}s)")
        results.append({"name": name, "features": cols, "n_feat": len(ORIG_COLS) + len(cols),
                        "oof_auc": auc, "fold_auc": fauc, "best_iters": iters})

    with pd.option_context("display.width", 200):
        print("\n" + "=" * 70)
        print(f"{'实验':<28}{'特征数':>7}{'OOF AUC':>12}{'Δ vs 基线':>14}")
        print("=" * 70)
        for r in results:
            delta = "—" if r["name"] == "baseline_orig" else f"{r['oof_auc'] - base_auc:+.6f}"
            print(f"{r['name']:<28}{r['n_feat']:>7}{r['oof_auc']:>12.6f}{delta:>14}")

    (OUT / "feature_batches.json").write_text(
        json.dumps({"baseline_oof_auc": base_auc, "params": B_PLUS, "seed": SEED, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n>>> 已写入 {OUT}/feature_batches.json")


if __name__ == "__main__":
    main()
