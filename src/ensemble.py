"""多模型集成：LGBM × 3 种子 + XGBoost + CatBoost。

全部使用 16 列特征集与同一套 5 折（seed 42），保证 OOF 可比。
每个模型跑完立即落盘，中断不丢结果。

用法:
  python -I src/ensemble.py                # 跑全部缺失的模型并融合
  python -I src/ensemble.py --models lgbm7 # 只跑指定模型
"""
from __future__ import annotations

import argparse
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
from baseline_lgbm import N_SPLITS, PARAMS, prepare  # noqa: E402
from feature_ablation import WEAK  # noqa: E402

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, Pool

    HAS_CAT = True
except ImportError:
    HAS_CAT = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"
ENS = OUT / "ensemble"
ENS.mkdir(parents=True, exist_ok=True)

FOLD_SEED = 42
LGB_BASE = {**PARAMS, "num_leaves": 24, "min_child_samples": 500,
            "reg_lambda": 10.0, "colsample_bytree": 0.5}
CAT_COLS = ["merchant_category", "auth_method"]


def fit_lgbm(X, y, folds, X_test, seed: int):
    params = {**LGB_BASE, "seed": seed}
    oof = np.zeros(len(X))
    tp = np.zeros(len(X_test))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / len(folds)
    return oof, tp


def fit_xgb(X, y, folds, X_test):
    oof = np.zeros(len(X))
    tp = np.zeros(len(X_test))
    for tr, va in folds:
        m = xgb.XGBClassifier(
            n_estimators=5000, learning_rate=0.03, max_depth=6, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0,
            tree_method="hist", enable_categorical=True, eval_metric="auc",
            early_stopping_rounds=100, n_jobs=-1, random_state=42,
        )
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / len(folds)
    return oof, tp


def fit_cat(X, y, folds, X_test):
    Xc, Xc_test = X.copy(), X_test.copy()
    for c in CAT_COLS:
        Xc[c] = Xc[c].astype(str)
        Xc_test[c] = Xc_test[c].astype(str)
    cat_idx = [Xc.columns.get_loc(c) for c in CAT_COLS]

    oof = np.zeros(len(X))
    tp = np.zeros(len(X_test))
    for tr, va in folds:
        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=10.0,
            loss_function="Logloss", eval_metric="AUC", random_seed=42,
            od_type="Iter", od_wait=100, verbose=False, thread_count=-1,
        )
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_idx), use_best_model=True)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        tp += m.predict_proba(Xc_test)[:, 1] / len(folds)
    return oof, tp


MODELS = {
    "lgbm42": ("lgbm", 42),
    "lgbm7": ("lgbm", 7),
    "lgbm2024": ("lgbm", 2024),
    "xgb": ("xgb", None),
    "cat": ("cat", None),
}


def load_saved(name: str):
    f = ENS / f"{name}_oof.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)["oof_pred"].to_numpy(), pd.read_csv(ENS / f"{name}_test.csv")["test_pred"].to_numpy()


def hill_climb(auc_fn, n: int, steps: int = 40):
    w = np.ones(n) / n
    best = auc_fn(w)
    for _ in range(steps):
        improved = False
        for i in range(n):
            for d in (0.05, -0.05, 0.02, -0.02):
                cand = w.copy()
                cand[i] = max(0.0, cand[i] + d)
                s = cand.sum()
                if s <= 0:
                    continue
                cand /= s
                v = auc_fn(cand)
                if v > best + 1e-9:
                    w, best, improved = cand, v, True
        if not improved:
            break
    return w, best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    X, y, X_test, _ = prepare()
    cols = [c for c in X.columns if c not in WEAK]
    X, X_test = X[cols], X_test[cols]
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(X, y))
    print(f"特征 {len(cols)} 列，折 seed {FOLD_SEED}，正类 {y.sum():,}\n")

    todo = args.models or list(MODELS)
    for name in todo:
        if load_saved(name) is not None:
            print(f"{name}: 已存在，跳过")
            continue
        kind, seed = MODELS[name]
        t0 = time.time()
        if kind == "lgbm":
            oof, tp = fit_lgbm(X, y, folds, X_test, seed)
        elif kind == "xgb":
            if not HAS_XGB:
                print("xgboost 不可用，跳过"); continue
            oof, tp = fit_xgb(X, y, folds, X_test)
        else:
            if not HAS_CAT:
                print("catboost 不可用，跳过"); continue
            oof, tp = fit_cat(X, y, folds, X_test)

        auc = roc_auc_score(y, oof)
        pd.DataFrame({"oof_pred": oof}).to_csv(ENS / f"{name}_oof.csv", index=False)
        pd.DataFrame({"test_pred": tp}).to_csv(ENS / f"{name}_test.csv", index=False)
        print(f"{name:<10} OOF {auc:.6f}  ({time.time() - t0:.0f}s)")

    # ---- 融合 ----
    names = [n for n in MODELS if load_saved(n) is not None]
    print(f"\n已保存模型: {names}")
    oofs = {n: load_saved(n)[0] for n in names}
    tests = {n: load_saved(n)[1] for n in names}

    print("\n单模型 / 模型间相关性:")
    print(f"{'模型':<10}{'OOF AUC':>10}")
    for n in names:
        print(f"{n:<10}{roc_auc_score(y, oofs[n]):>10.6f}")
    corr = pd.DataFrame(oofs).corr(method="spearman").round(4)
    print("\nOOF Spearman 相关性:")
    print(corr.to_string())

    M = np.column_stack([oofs[n] for n in names])
    Mt = np.column_stack([tests[n] for n in names])

    equal_auc = roc_auc_score(y, M.mean(axis=1))
    rank_auc = roc_auc_score(y, pd.DataFrame(M).rank(pct=True).mean(axis=1))
    w, w_auc = hill_climb(lambda w: roc_auc_score(y, M @ w), len(names))

    print("\n" + "=" * 60)
    print(f"{'融合方式':<24}{'OOF AUC':>12}")
    print("=" * 60)
    print(f"{'等权平均':<24}{equal_auc:>12.6f}")
    print(f"{'Rank 平均':<24}{rank_auc:>12.6f}")
    print(f"{'OOF 优化权重':<24}{w_auc:>12.6f}")
    print("\n优化权重:", dict(zip(names, np.round(w, 4))))

    pd.DataFrame({"oof_pred": M.mean(axis=1)}).to_csv(ENS / "blend_equal_oof.csv", index=False)
    pd.DataFrame({"test_pred": Mt.mean(axis=1)}).to_csv(ENS / "blend_equal_test.csv", index=False)
    pd.DataFrame({"oof_pred": M @ w}).to_csv(ENS / "blend_weighted_oof.csv", index=False)
    pd.DataFrame({"test_pred": Mt @ w}).to_csv(ENS / "blend_weighted_test.csv", index=False)

    summary = {
        "features": cols, "fold_seed": FOLD_SEED,
        "single": {n: float(roc_auc_score(y, oofs[n])) for n in names},
        "spearman": corr.to_dict(),
        "blend": {"equal": float(equal_auc), "rank": float(rank_auc),
                  "weighted": float(w_auc), "weights": dict(zip(names, w.tolist()))},
    }
    (OUT / "ensemble.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>>> 已写入 {OUT}/ensemble.json 与 {ENS}/")


if __name__ == "__main__":
    main()
