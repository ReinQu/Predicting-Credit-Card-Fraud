"""把 test 预测对齐到 sample_submission 并校验。

用法:
  python -I src/make_submission.py --pred reports/test_pred_lgbm_baseline.csv \
      --name 2026-11-07_lgbm_baseline_cv0.945650
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e11"
SUB_DIR = ROOT / "submissions"
ID = "id"


def build(sample: pd.DataFrame, pred_path: Path, pred_col: str) -> pd.DataFrame:
    preds = pd.read_csv(pred_path)
    assert set([ID, pred_col]) <= set(preds.columns), f"{pred_path} 缺少列 {ID}/{pred_col}"
    assert preds[ID].is_unique, "预测文件的 id 有重复"

    s = pd.Series(preds[pred_col].to_numpy(), index=pd.Index(preds[ID], name=ID))
    target = [c for c in sample.columns if c != ID][0]

    sub = sample.copy()
    sub[target] = s.reindex(sample[ID]).to_numpy()
    return sub


def validate(sub: pd.DataFrame, sample: pd.DataFrame, target: str) -> None:
    assert list(sub.columns) == list(sample.columns), "列名或列顺序不一致"
    assert len(sub) == len(sample), f"行数不一致: {len(sub)} vs {len(sample)}"
    assert sub[ID].is_unique, "id 有重复"
    assert set(sub[ID]) == set(sample[ID]), "id 集合与 sample 不一致"
    assert sub[target].notna().all(), "预测值含 NaN"
    assert pd.api.types.is_numeric_dtype(sub[target]), "预测列不是数值"
    assert sub[target].between(0, 1).all(), "概率超出 [0,1]"
    print(f"rows: {len(sub)}  range: [{sub[target].min():.8f}, {sub[target].max():.8f}]")
    print(f"mean: {sub[target].mean():.8f}  (train 正类率参考 0.0167)")
    print(sub.head())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True, help="不含扩展名的文件名")
    ap.add_argument("--pred-col", default=None)
    args = ap.parse_args()

    sample = pd.read_csv(DATA / "sample_submission.csv")
    target = [c for c in sample.columns if c != ID][0]
    pred_path = Path(args.pred)
    pred_col = args.pred_col or [c for c in pd.read_csv(pred_path, nrows=0).columns if c != ID][0]

    sub = build(sample, pred_path, pred_col)
    validate(sub, sample, target)

    SUB_DIR.mkdir(exist_ok=True)
    out = SUB_DIR / f"{args.name}.csv"
    assert not out.exists(), f"{out} 已存在，不覆盖历史提交"
    sub.to_csv(out, index=False)
    print(f"\n>>> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
