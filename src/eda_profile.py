"""数据画像：结构 / 缺失 / 目标 / 分布漂移 / 泄漏检查。

用法: python -I src/eda_profile.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

TARGET = "is_fraud"
ID = "id"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    train, test = load()
    report: dict = {}

    section("1. 规模与结构")
    print(f"train: {train.shape}   test: {test.shape}")
    print(f"train 列数 {train.shape[1]}, test 列数 {test.shape[1]}")
    tr_feats = set(train.columns) - {TARGET}
    te_feats = set(test.columns) - {ID}
    print(f"特征列一致: {tr_feats - {ID} == te_feats}")
    print("\ntrain 比 test 多的列:", sorted(set(train.columns) - set(test.columns)))
    print("test 比 train 多的列:", sorted(set(test.columns) - set(train.columns)))
    report["shape"] = {"train": list(train.shape), "test": list(test.shape)}

    section("2. dtypes / 唯一值 / 缺失")
    cols = [c for c in train.columns if c in test.columns]
    profile = pd.DataFrame(
        {
            "dtype": train.dtypes.astype(str),
            "nunique": train.nunique(),
            "null_train": train[cols].isna().mean().round(4),
            "null_test": test[cols].isna().mean().round(4),
        }
    )
    print(profile.to_string())
    report["null_total"] = float(profile["null_train"].sum() + profile["null_test"].sum())

    section("3. 目标分布")
    vc = train[TARGET].value_counts().sort_index()
    print(vc.to_string())
    rate = train[TARGET].mean()
    print(f"\n正类率: {rate:.6f}  ({vc.get(1.0, 0):,} / {len(train):,})")
    print(f"不平衡比: 1 : {1 / rate:.1f}")
    report["target"] = {"rate": float(rate), "counts": {str(k): int(v) for k, v in vc.items()}}

    section("4. 数值特征")
    num = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in (ID, TARGET)]
    desc = train[num].describe().T
    desc["skew"] = train[num].skew()
    desc["nunique"] = train[num].nunique()
    print(desc.round(3).to_string())
    report["numeric"] = num

    section("5. 类别特征")
    cat = [c for c in train.columns if c not in num and c not in (ID, TARGET)]
    for c in cat:
        vc = train[c].value_counts()
        print(f"\n[{c}] 基数={train[c].nunique()}  类型={train[c].dtype}")
        print(vc.head(12).to_string())
        print(f"  目标率: {train.groupby(c, observed=True)[TARGET].mean().round(5).to_dict()}")
    print("\n布尔列（True/False）检测:")
    for c in cat:
        if train[c].dtype == object and set(train[c].dropna().unique()) <= {"True", "False"}:
            print(f"  {c}: 需要转 0/1")
    report["categorical"] = {c: int(train[c].nunique()) for c in cat}

    section("6. train/test 分布对比")
    drift = []
    for c in num:
        mt, ms = train[c].mean(), test[c].mean()
        pooled = np.sqrt((train[c].std() ** 2 + test[c].std() ** 2) / 2)
        drift.append((c, mt, ms, abs(mt - ms) / (pooled + 1e-12)))
    d = pd.DataFrame(drift, columns=["col", "train_mean", "test_mean", "smd"]).sort_values("smd", ascending=False)
    print(d.round(5).to_string(index=False))

    print("\n类别特征 TV 距离（总变差，0=完全一致）:")
    tv = {}
    for c in cat:
        p = train[c].value_counts(normalize=True)
        q = test[c].value_counts(normalize=True)
        idx = p.index.union(q.index)
        tv[c] = float(0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())
        print(f"  {c}: {tv[c]:.5f}")
    report["drift"] = {"max_smd": float(d.iloc[0]["smd"]), "max_smd_col": str(d.iloc[0]["col"]), "tv": tv}

    section("7. 重复 / 泄漏检查")
    feat = [c for c in cols if c != ID]
    dup_tr = train.duplicated(subset=feat).sum()
    dup_te = test.duplicated(subset=feat).sum()
    print(f"train 特征完全重复行: {dup_tr:,}")
    print(f"test  特征完全重复行: {dup_te:,}")

    merged = train[feat].merge(test[feat], how="inner", on=feat)
    print(f"train/test 特征重叠行数: {len(merged):,}")

    print(f"\nID 范围: train [{train[ID].min()}, {train[ID].max()}]  test [{test[ID].min()}, {test[ID].max()}]")
    print(f"ID 是否有交集: {len(set(train[ID]) & set(test[ID]))}")
    print(f"ID 与目标相关性 (point-biserial): {np.corrcoef(train[ID], train[TARGET])[0, 1]:.6f}")
    print(f"ID 是否连续: {train[ID].is_monotonic_increasing and (train[ID].diff().dropna() == 1).all()}")

    # id 分箱后的目标率，看是否存在随时间漂移
    train["_id_bin"] = pd.qcut(train[ID], 10, labels=False, duplicates="drop")
    print("\n按 ID 十等分的正类率（看是否有时间漂移）:")
    print(train.groupby("_id_bin")[TARGET].mean().round(5).to_string())
    report["id_target_corr"] = float(np.corrcoef(train[ID], train[TARGET])[0, 1])

    section("8. 与目标的关系")
    corr = train[num + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("数值特征与目标的相关性:")
    print(corr.round(4).to_string())
    report["top_corr"] = corr.head(8).round(4).to_dict()

    print("\n数值特征之间的高相关对 (|r| > 0.8):")
    cm = train[num].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(cm, 0)
    cm = pd.DataFrame(cm, index=num, columns=num)
    pairs = [(a, b, cm.loc[a, b]) for a in cm.index for b in cm.columns if a < b and cm.loc[a, b] > 0.8]
    print("  " + ("\n  ".join(f"{a} ~ {b}: {v:.3f}" for a, b, v in pairs) if pairs else "无"))

    section("9. 单特征区分力（AUC，粗筛）")
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for c in num:
        aucs[c] = roc_auc_score(train[TARGET], train[c])
        aucs[c] = max(aucs[c], 1 - aucs[c])
    for c in cat:
        aucs[c] = roc_auc_score(train[TARGET], train[c].astype("category").cat.codes)
        aucs[c] = max(aucs[c], 1 - aucs[c])
    s = pd.Series(aucs).sort_values(ascending=False)
    print(s.round(4).to_string())
    report["single_feature_auc"] = s.round(4).to_dict()

    (OUT / "eda_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n>>> 已写入 {OUT / 'eda_profile.json'}")


if __name__ == "__main__":
    main()
