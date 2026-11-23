"""LightGBM 基线：5 折 StratifiedKFold + OOF AUC。

产物：
  reports/cv_lgbm_baseline.json     每折指标 / 总体指标 / 折划分索引
  reports/oof_lgbm_baseline.csv     OOF 预测（后续特征选择、融合都以它为准）
  reports/test_pred_lgbm_baseline.csv  各折 test 预测的均值（后续生成提交时复用）
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"

TARGET = "is_fraud"
ID = "id"
SEED = 42
N_SPLITS = 5

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "n_jobs": -1,
    "verbosity": -1,
    "seed": SEED,
}


def prepare() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    y = train[TARGET].astype(int).to_numpy()
    X = train.drop(columns=[ID, TARGET])
    X_test = test.drop(columns=[ID])

    # bool 在 sklearn 系里行为不一致，统一转成 int8
    bool_cols = [c for c in X.columns if X[c].dtype == bool]
    for c in bool_cols:
        X[c] = X[c].astype("int8")
        X_test[c] = X_test[c].astype("int8")

    # 类别列必须用同一套 categories，否则 test 里没出现过的类别会变成 NaN
    # 注意 pandas 3.x 的字符串列 dtype 是 str/StringDtype，不是 object
    cat_cols = [c for c in X.columns if pd.api.types.is_string_dtype(X[c])]
    for c in cat_cols:
        cats = sorted(set(X[c]) | set(X_test[c]))
        X[c] = pd.Categorical(X[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)

    return X, y, X_test, cat_cols


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    X, y, X_test, cat_cols = prepare()
    print(f"X: {X.shape}  X_test: {X_test.shape}  positives: {y.sum():,} ({y.mean():.4%})")
    print(f"categorical: {cat_cols}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_auc, best_iters, fold_idx = [], [], []
    importances = []

    for k, (tr, va) in enumerate(splits, 1):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
        auc = roc_auc_score(y[va], oof[va])
        fold_auc.append(auc)
        best_iters.append(int(model.best_iteration_ or PARAMS["n_estimators"]))
        fold_idx.append({"fold": k, "n_train": len(tr), "n_valid": len(va), "best_iter": best_iters[-1]})
        importances.append(pd.Series(model.feature_importances_, index=X.columns))
        print(f"fold {k}: AUC {auc:.6f}  best_iter {best_iters[-1]}  ({time.time() - t0:.0f}s)")

    overall = roc_auc_score(y, oof)
    print(f"\nOOF AUC: {overall:.6f}")
    print(f"每折均值: {np.mean(fold_auc):.6f} ± {np.std(fold_auc):.6f}")
    print(f"总耗时: {time.time() - t0:.0f}s")

    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\n特征重要性 (被用作分裂特征的次数, 5 折均值, top 20):")
    print(imp.head(20).round(1).to_string())

    result = {
        "model": "lgbm_baseline",
        "params": {k: v for k, v in PARAMS.items()},
        "n_splits": N_SPLITS,
        "seed": SEED,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "oof_auc": float(overall),
        "fold_auc": [float(a) for a in fold_auc],
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc)),
        "folds": fold_idx,
        "best_iters": best_iters,
        "feature_importance": imp.round(2).to_dict(),
        "note": "早停使用验证折，OOF 可能略微乐观；后续做正式对比时改为固定轮数或内层验证。",
    }
    (OUT / "cv_lgbm_baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame({ID: pd.read_csv(DATA / "train.csv", usecols=[ID])[ID], "oof_pred": oof}).to_csv(
        OUT / "oof_lgbm_baseline.csv", index=False
    )
    pd.DataFrame({ID: pd.read_csv(DATA / "test.csv", usecols=[ID])[ID], "test_pred": test_pred}).to_csv(
        OUT / "test_pred_lgbm_baseline.csv", index=False
    )
    print(f"\n>>> 已写入 {OUT}/cv_lgbm_baseline.json, oof_lgbm_baseline.csv, test_pred_lgbm_baseline.csv")


if __name__ == "__main__":
    main()
