"""特征消融：留一法删除 8 个弱信号列，看各自对 CV 的边际贡献。

基线 = B_plus 参数 + 原始 24 列，seed 42，OOF 0.946789。
噪声水平实测 ~0.00015，因此 |Δ| < 0.0003 视为无差异。

用法: python -I src/feature_ablation.py
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
from baseline_lgbm import N_SPLITS, PARAMS, SEED, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"

B_PLUS = {
    **PARAMS,
    "num_leaves": 24,
    "min_child_samples": 500,
    "reg_lambda": 10.0,
    "colsample_bytree": 0.5,
}

# 单特征 AUC ≈ 0.50–0.51 的列
WEAK = [
    "card_type",
    "device_type",
    "channel",
    "distance_from_home_km",
    "card_age_months",
    "time_of_day_hour",
    "day_of_week",
    "account_balance_usd",
]


def run(cols: list[str], X, y, folds) -> tuple[float, list[float], list[int]]:
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
    X, y, X_test, _ = prepare()
    all_cols = list(X.columns)
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

    exprs = [("baseline_24", all_cols), ("drop_all8", [c for c in all_cols if c not in WEAK])]
    exprs += [(f"drop_{c}", [x for x in all_cols if x != c]) for c in WEAK]

    results = []
    base_auc = None
    for name, cols in exprs:
        t0 = time.time()
        auc, fauc, iters = run(cols, X, y, folds)
        if base_auc is None:
            base_auc = auc
        delta = auc - base_auc
        print(f"{name:<32} n={len(cols):<3} OOF {auc:.6f}  Δ {delta:+.6f}  ({time.time() - t0:.0f}s)")
        results.append({"name": name, "n_features": len(cols), "oof_auc": auc,
                        "delta": delta, "fold_auc": fauc, "best_iters": iters})

    print("\n" + "=" * 66)
    print(f"{'实验':<32}{'特征数':>7}{'OOF AUC':>12}{'Δ':>12}")
    print("=" * 66)
    for r in sorted(results, key=lambda r: -r["oof_auc"]):
        print(f"{r['name']:<32}{r['n_features']:>7}{r['oof_auc']:>12.6f}{r['delta']:>+12.6f}")
    print("\n噪声水平 ~0.00015；|Δ| < 0.0003 视为无差异。")

    (OUT / "feature_ablation.json").write_text(
        json.dumps({"baseline_oof_auc": base_auc, "weak_cols": WEAK, "seed": SEED, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f">>> 已写入 {OUT}/feature_ablation.json")


if __name__ == "__main__":
    main()
