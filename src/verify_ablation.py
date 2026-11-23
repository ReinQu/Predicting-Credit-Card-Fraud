"""在第二个种子上验证消融结果，并测试"保留 channel"的变体。

用法: python -I src/verify_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_lgbm import N_SPLITS, prepare  # noqa: E402
from feature_ablation import WEAK, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
SEEDS = [42, 2024]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    X, y, X_test, _ = prepare()
    all_cols = list(X.columns)

    drop7 = [c for c in WEAK if c != "channel"]
    configs = {
        "baseline_24": all_cols,
        "drop_all8": [c for c in all_cols if c not in WEAK],
        "drop7_keep_channel": [c for c in all_cols if c not in drop7],
    }

    results: dict = {}
    for seed in SEEDS:
        folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X, y))
        for name, cols in configs.items():
            auc, fauc, iters = run(cols, X, y, folds)
            print(f"seed {seed}  {name:<22} n={len(cols):<3} OOF {auc:.6f}  iters {iters}")
            results.setdefault(name, {})[seed] = {"n_features": len(cols), "oof_auc": auc,
                                                  "fold_auc": fauc, "best_iters": iters}

    print("\n" + "=" * 72)
    print(f"{'配置':<24}{'seed42':>10}{'seed2024':>11}{'均值':>11}{'vs baseline':>14}")
    print("=" * 72)
    base = np.mean([results["baseline_24"][s]["oof_auc"] for s in SEEDS])
    for name in configs:
        v = [results[name][s]["oof_auc"] for s in SEEDS]
        print(f"{name:<24}{v[0]:>10.6f}{v[1]:>11.6f}{np.mean(v):>11.6f}{np.mean(v) - base:>+14.6f}")

    (OUT / "verify_ablation.json").write_text(
        json.dumps({"seeds": SEEDS, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n>>> 已写入 {OUT}/verify_ablation.json")


if __name__ == "__main__":
    main()
