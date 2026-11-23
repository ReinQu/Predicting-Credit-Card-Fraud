"""用两个种子验证 B 的正则化方向，并沿该方向再推一组。

对照：
  baseline  原始参数
  B_reg     fold-seed 42 上的当前最佳（leaves32/mcs300/λ5/col0.6）
  B_plus    继续加强正则（leaves24/mcs500/λ10/col0.5）

每个配置在 seed 42 和 seed 2024 两套切分上各跑一次，看提升是否稳定。

用法: python -I src/verify_params.py
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
from baseline_lgbm import ID, N_SPLITS, PARAMS, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"

BASE = dict(PARAMS)

CONFIGS: dict[str, dict] = {
    "baseline": {"params": BASE, "early_stopping": 100},
    "B_reg": {
        "params": {**BASE, "num_leaves": 32, "min_child_samples": 300, "reg_lambda": 5.0, "colsample_bytree": 0.6},
        "early_stopping": 100,
    },
    "B_plus": {
        "params": {**BASE, "num_leaves": 24, "min_child_samples": 500, "reg_lambda": 10.0, "colsample_bytree": 0.5},
        "early_stopping": 100,
    },
}

SEEDS = [42, 2024]


def run(name: str, cfg: dict, X, y, folds, X_test) -> dict:
    t0 = time.time()
    params = cfg["params"]
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_auc, best_iters = [], []

    for tr, va in folds:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / len(folds)
        fold_auc.append(float(roc_auc_score(y[va], oof[va])))
        best_iters.append(int(model.best_iteration_ or params["n_estimators"]))

    return {
        "name": name,
        "oof_auc": float(roc_auc_score(y, oof)),
        "fold_auc": fold_auc,
        "fold_auc_std": float(np.std(fold_auc)),
        "best_iters": best_iters,
        "seconds": round(time.time() - t0, 1),
        "oof": oof,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    X, y, X_test, _ = prepare()

    results = {}
    for seed in SEEDS:
        folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X, y))
        for name, cfg in CONFIGS.items():
            r = run(name, cfg, X, y, folds, X_test)
            print(f"seed {seed}  {name:10s} OOF {r['oof_auc']:.6f}  std {r['fold_auc_std']:.6f}  "
                  f"iters {r['best_iters']}  ({r['seconds']:.0f}s)")
            results.setdefault(name, {})[seed] = {k: v for k, v in r.items() if k != "oof"}
            if name == "B_plus":
                pd.DataFrame({ID: pd.read_csv(DATA / "train.csv", usecols=[ID])[ID], "oof_pred": r["oof"]}).to_csv(
                    OUT / f"oof_seed{seed}_B_plus.csv", index=False
                )

    print("\n" + "=" * 74)
    print(f"{'配置':<12}{'seed42':>10}{'seed2024':>12}{'两seed均值':>13}{'vs baseline':>14}")
    print("=" * 74)
    base_mean = np.mean([results["baseline"][s]["oof_auc"] for s in SEEDS])
    for name in CONFIGS:
        vals = [results[name][s]["oof_auc"] for s in SEEDS]
        print(f"{name:<12}{vals[0]:>10.6f}{vals[1]:>12.6f}{np.mean(vals):>13.6f}{np.mean(vals) - base_mean:>+14.6f}")

    (OUT / "verify_params.json").write_text(
        json.dumps({"seeds": SEEDS, "n_splits": N_SPLITS, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n>>> 已写入 {OUT}/verify_params.json")


if __name__ == "__main__":
    main()
