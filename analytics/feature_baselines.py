"""Lightweight baseline probes for alpha/zeta predictability from summary features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils_io import (
    balanced_file_subset,
    discover_data_files,
    ensure_dir,
    inspect_data_file,
    iter_dataset_samples,
    save_json,
)
from utils_plots import plot_baseline_results
from utils_stats import grouped_regression_metrics, sample_feature_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("forensics_outputs"))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Per-file sample cap")
    parser.add_argument("--sample_stride", type=int, default=1, help="Temporal stride when reading trajectories")
    parser.add_argument("--quick_mode", action="store_true")
    return parser.parse_args()


def build_feature_table(
    data_root: Path,
    out_dir: Path,
    max_files: Optional[int] = None,
    max_samples: Optional[int] = None,
    sample_stride: int = 1,
    quick_mode: bool = False,
) -> pd.DataFrame:
    """Iterate local samples and save hand-crafted feature summaries."""
    baseline_dir = ensure_dir(out_dir / "baselines")
    cache_path = baseline_dir / "trajectory_features.csv"
    if cache_path.exists() and not quick_mode:
        return pd.read_csv(cache_path)

    file_limit = max_files if max_files is not None else (10 if quick_mode else None)
    sample_limit = max_samples if max_samples is not None else (2 if quick_mode else None)
    files = discover_data_files(data_root)
    if file_limit is not None:
        files = balanced_file_subset(files, data_root, file_limit)
    metadata_map = {str(path.resolve()): inspect_data_file(path, data_root=data_root) for path in files}

    rows: List[Dict[str, object]] = []
    sample_counter = 0
    for sample in tqdm(
        iter_dataset_samples(files, metadata_map, max_samples_per_file=sample_limit, sample_stride=sample_stride),
        desc="Extracting baseline features",
    ):
        array = sample["array"]
        channel_names = sample["channel_names"]
        row = sample_feature_summary(array, channel_names)
        row.update(
            {
                "sample_id": sample_counter,
                "file_path": sample["file_path"],
                "split": sample["split"],
                "alpha": sample["labels"].get("alpha"),
                "zeta": sample["labels"].get("zeta"),
                "sample_index": sample["sample_index"],
            }
        )
        rows.append(row)
        sample_counter += 1

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def run_regression_baselines(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped CV on simple regressors to test label recoverability."""
    feature_cols = [
        col for col in feature_df.columns
        if col not in {"sample_id", "file_path", "split", "alpha", "zeta", "sample_index"}
    ]
    groups = feature_df["file_path"]
    regressors = {
        "linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "knn": Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "rf": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    }
    n_splits = min(5, feature_df["file_path"].nunique())
    splitter = GroupKFold(n_splits=max(2, n_splits))
    rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    X = feature_df[feature_cols].to_numpy(dtype=float)

    for target in ("alpha", "zeta"):
        y = feature_df[target].to_numpy(dtype=float)
        for model_name, model in regressors.items():
            for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                rows.append(grouped_regression_metrics(y[test_idx], pred, target_name=target, model_name=model_name, fold=fold))
                for idx, pred_value in zip(test_idx, pred):
                    pred_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "fold": fold,
                            "sample_id": int(feature_df.iloc[idx]["sample_id"]),
                            "y_true": float(y[idx]),
                            "y_pred": float(pred_value),
                        }
                    )

    results_df = pd.DataFrame(rows)
    preds_df = pd.DataFrame(pred_rows)
    return results_df, preds_df


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    baseline_dir = ensure_dir(out_dir / "baselines")

    feature_df = build_feature_table(
        data_root=args.data_root,
        out_dir=out_dir,
        max_files=args.max_files,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        quick_mode=args.quick_mode,
    )
    feature_df.to_csv(baseline_dir / "trajectory_features.csv", index=False)

    results_df, preds_df = run_regression_baselines(feature_df)
    summary_df = results_df.groupby(["target", "model"], as_index=False).agg({"rmse": "mean", "mae": "mean", "r2": "mean"})
    results_df.to_csv(baseline_dir / "baseline_cv_results.csv", index=False)
    summary_df.to_csv(baseline_dir / "baseline_summary.csv", index=False)
    preds_df.to_csv(baseline_dir / "baseline_predictions.csv", index=False)
    save_json(
        baseline_dir / "baseline_summary.json",
        {
            "n_samples": int(len(feature_df)),
            "n_unique_files": int(feature_df["file_path"].nunique()),
            "summary": summary_df.to_dict(orient="records"),
        },
    )
    plot_baseline_results(summary_df, baseline_dir / "baseline_results.png")
    print(f"Saved baseline outputs to {baseline_dir}")


if __name__ == "__main__":
    main()
