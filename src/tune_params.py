"""参数诊断实验：同一套 5 折切分下比较 3 组参数。

这不是网格搜索，是三条互斥假设的对照实验：
  A 容量+    -> 模型是否欠拟合
  B 正则+    -> 模型是否过拟合
  C 加权     -> 1:60 不平衡是否值得处理

用法: python -I src/tune_params.py
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
from baseline_lgbm import ID, N_SPLITS, PARAMS, SEED, TARGET, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"

BASE = {k: v for k, v in PARAMS.items()}

EXPERIMENTS: dict[str, dict] = {
    "A_capacity": {
        "params": {**BASE, "learning_rate": 0.02, "num_leaves": 128, "min_child_samples": 50, "n_estimators": 8000},
        "early_stopping": 200,
        "hypothesis": "模型在欠拟合，加大容量能提升",
    },
    "B_regularized": {
        "params": {**BASE, "num_leaves": 32, "min_child_samples": 300, "reg_lambda": 5.0, "colsample_bytree": 0.6},
        "early_stopping": 100,
        "hypothesis": "模型在过拟合噪声，加强正则能提升",
    },
    "C_weighted": {
        "params": {**BASE, "scale_pos_weight": 59},
        "early_stopping": 100,
        "hypothesis": "处理类不平衡能提升（预期几乎无效）",
    },
}


def run_one(name: str, cfg: dict, X, y, folds, X_test) -> dict:
    t0 = time.time()
    params = cfg["params"]
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_auc, best_iters = [], []

    for k, (tr, va) in enumerate(folds, 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(cfg["early_stopping"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / len(folds)
        fold_auc.append(float(roc_auc_score(y[va], oof[va])))
        best_iters.append(int(model.best_iteration_ or params["n_estimators"]))

    overall = float(roc_auc_score(y, oof))
    print(
        f"{name:14s} OOF {overall:.6f}  std {np.std(fold_auc):.6f}  iters {best_iters}  "
        f"({time.time() - t0:.0f}s)"
    )
    pd.DataFrame({ID: pd.read_csv(DATA / "train.csv", usecols=[ID])[ID], "oof_pred": oof}).to_csv(
        OUT / f"oof_tune_{name}.csv", index=False
    )
    return {
        "name": name,
        "hypothesis": cfg["hypothesis"],
        "params": params,
        "early_stopping": cfg["early_stopping"],
        "fold_auc": fold_auc,
        "oof_auc": overall,
        "fold_auc_std": float(np.std(fold_auc)),
        "best_iters": best_iters,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    X, y, X_test, _ = prepare()
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
    print(f"folds fixed: {N_SPLITS} x seed {SEED}\n")

    baseline_auc = 0.945650
    results = [run_one(n, c, X, y, folds, X_test) for n, c in EXPERIMENTS.items()]

    print("\n" + "=" * 72)
    print(f"{'实验':<16}{'OOF AUC':>10}{'vs 基线':>12}{'折间 std':>11}")
    print("=" * 72)
    print(f"{'基线':<16}{baseline_auc:>10.6f}{'—':>12}")
    for r in results:
        print(f"{r['name']:<16}{r['oof_auc']:>10.6f}{r['oof_auc'] - baseline_auc:>+12.6f}{r['fold_auc_std']:>11.6f}")

    payload = {"baseline_oof_auc": baseline_auc, "seed": SEED, "n_splits": N_SPLITS, "results": results}
    (OUT / "tune_params.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>>> 已写入 {OUT}/tune_params.json")


if __name__ == "__main__":
    main()
