"""Complete local analytics pipeline for the active matter dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_baselines import build_feature_table, run_regression_baselines
from utils_io import (
    balanced_file_subset,
    discover_data_files,
    ensure_dir,
    inspect_data_file,
    iter_dataset_samples,
    save_json,
    summarize_layouts,
)
from utils_plots import (
    plot_channel_histograms,
    plot_correlation_heatmap,
    plot_example_frames,
    plot_label_distributions,
    plot_pca_embedding,
    plot_regime_heatmaps,
    plot_spectra,
    plot_temporal_summary,
)
from utils_stats import (
    cross_channel_correlation,
    duplicate_suspicion_scores,
    pca_embedding,
    per_channel_stats,
    safe_corrcoef,
    spectral_metrics,
    temporal_metrics,
)


COURSE_BRIEF = {
    "timesteps": 81,
    "height": 256,
    "width": 256,
    "processed_clip_length": 16,
    "processed_crop": 224,
    "channels": 11,
    "alpha_zeta_combinations": 45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("forensics_outputs"))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Per-file logical sample cap")
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--quick_mode", action="store_true")
    parser.add_argument("--run_baselines", action="store_true")
    return parser.parse_args()


def collect_sample_analytics(
    files: Sequence[Path],
    metadata_map: Dict[str, object],
    max_samples_per_file: Optional[int],
    sample_stride: int,
    out_dir: Path,
    skip_plots: bool,
) -> Dict[str, pd.DataFrame]:
    """Iterate samples once and collect major analytics tables."""
    plot_dir = ensure_dir(out_dir / "plots")
    table_dir = ensure_dir(out_dir / "tables")

    global_stat_rows: List[pd.DataFrame] = []
    sample_rows: List[Dict[str, object]] = []
    autocorr_rows: List[pd.DataFrame] = []
    smoothness_rows: List[pd.DataFrame] = []
    spectral_rows: List[pd.DataFrame] = []
    radial_spectra: List[Dict[str, np.ndarray]] = []
    flat_samples: List[np.ndarray] = []
    histogram_frames: List[pd.DataFrame] = []
    example_done = False

    sample_iter = iter_dataset_samples(
        files=files,
        metadata_map=metadata_map,
        max_files=None,
        max_samples_per_file=max_samples_per_file,
        sample_stride=sample_stride,
    )
    for sample_id, sample in enumerate(tqdm(sample_iter, desc="Analyzing samples")):
        array = sample["array"]
        channel_names = sample["channel_names"]
        labels = sample["labels"]

        stats_df = per_channel_stats(array, channel_names)
        stats_df["sample_id"] = sample_id
        stats_df["file_path"] = sample["file_path"]
        stats_df["split"] = sample["split"]
        stats_df["alpha"] = labels.get("alpha")
        stats_df["zeta"] = labels.get("zeta")
        global_stat_rows.append(stats_df)

        temporal = temporal_metrics(array, channel_names)
        smooth_df = temporal["smoothness"].copy()
        smooth_df["sample_id"] = sample_id
        smooth_df["file_path"] = sample["file_path"]
        smooth_df["alpha"] = labels.get("alpha")
        smooth_df["zeta"] = labels.get("zeta")
        smoothness_rows.append(smooth_df)
        ac_df = temporal["autocorr"].copy()
        ac_df["sample_id"] = sample_id
        autocorr_rows.append(ac_df)

        spectral = spectral_metrics(array, channel_names)
        per_channel = spectral["per_channel"].copy()
        per_channel["sample_id"] = sample_id
        per_channel["file_path"] = sample["file_path"]
        per_channel["alpha"] = labels.get("alpha")
        per_channel["zeta"] = labels.get("zeta")
        spectral_rows.append(per_channel)
        if sample_id < 4:
            radial_spectra.extend(spectral["spectra"])

        corr = cross_channel_correlation(array, channel_names)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "file_path": sample["file_path"],
                "split": sample["split"],
                "alpha": labels.get("alpha"),
                "zeta": labels.get("zeta"),
                "n_frames": int(array.shape[0]),
                "height": int(array.shape[1]),
                "width": int(array.shape[2]),
                "n_channels": int(array.shape[3]),
                **temporal["scalar_summary"],
                **spectral["scalar_summary"],
                **corr["scalar_summary"],
            }
        )

        flat = array.reshape(-1, array.shape[-1])
        if flat.shape[0] > 50000:
            idx = np.linspace(0, flat.shape[0] - 1, 50000, dtype=np.int64)
            flat = flat[idx]
        flat_samples.append(flat)

        if sample_id < 3:
            idx = np.linspace(0, flat.shape[0] - 1, min(25000, flat.shape[0]), dtype=np.int64)
            histogram_frames.append(pd.DataFrame(flat[idx], columns=channel_names))

        if not skip_plots and not example_done:
            plot_example_frames(
                array=np.moveaxis(array, -1, 0).transpose(1, 2, 3, 0),
                channel_names=channel_names,
                out_path=plot_dir / "example_frames.png",
                title="Example raw trajectory frames",
            )
            example_done = True

    channel_stats_df = pd.concat(global_stat_rows, ignore_index=True)
    sample_summary_df = pd.DataFrame(sample_rows)
    smoothness_df = pd.concat(smoothness_rows, ignore_index=True)
    autocorr_df = pd.concat(autocorr_rows, ignore_index=True)
    spectral_df = pd.concat(spectral_rows, ignore_index=True)
    flat_matrix = np.concatenate(flat_samples, axis=0)
    channel_corr = safe_corrcoef(flat_matrix)

    channel_stats_df.to_csv(table_dir / "per_sample_channel_stats.csv", index=False)
    sample_summary_df.to_csv(table_dir / "sample_summary.csv", index=False)
    smoothness_df.to_csv(table_dir / "temporal_smoothness.csv", index=False)
    autocorr_df.to_csv(table_dir / "temporal_autocorr.csv", index=False)
    spectral_df.to_csv(table_dir / "spectral_metrics.csv", index=False)
    pd.DataFrame(channel_corr).to_csv(table_dir / "channel_correlation_matrix.csv", index=False)

    if not skip_plots:
        plot_correlation_heatmap(channel_corr, channel_names, plot_dir / "channel_correlation_heatmap.png")
        plot_temporal_summary(
            smoothness_df.groupby("channel", as_index=False).mean(numeric_only=True),
            autocorr_df.groupby(["channel", "lag"], as_index=False).mean(numeric_only=True),
            plot_dir,
        )
        plot_spectra(radial_spectra[: len(channel_names)], plot_dir / "radial_power_spectra.png")
        hist_frame = pd.concat(histogram_frames, ignore_index=True)
        plot_channel_histograms(channel_stats_df, hist_frame, plot_dir / "channel_histograms.png")

    return {
        "channel_stats": channel_stats_df,
        "sample_summary": sample_summary_df,
        "smoothness": smoothness_df,
        "autocorr": autocorr_df,
        "spectral": spectral_df,
        "channel_corr": pd.DataFrame(channel_corr, index=channel_names, columns=channel_names),
    }


def compare_to_brief(metadata_map: Dict[str, object]) -> Dict[str, object]:
    """Verify local structure against course-brief expectations."""
    metadata_list = list(metadata_map.values())
    summary = summarize_layouts(metadata_list)
    first = metadata_list[0]
    first_field = first.fields[0] if first.fields else None

    observed = {
        "layout": first.layout,
        "n_timesteps": first_field.shape[1] if first_field and len(first_field.shape) >= 2 else None,
        "height": first_field.shape[2] if first_field and len(first_field.shape) >= 3 else None,
        "width": first_field.shape[3] if first_field and len(first_field.shape) >= 4 else None,
        "n_channels": len(first.detected_channel_names),
        "alpha_zeta_combinations": summary["num_alpha_zeta_combinations"],
    }
    return {
        "course_brief": COURSE_BRIEF,
        "observed_local_copy": observed,
        "matches": {
            "timesteps": observed["n_timesteps"] == COURSE_BRIEF["timesteps"],
            "height": observed["height"] == COURSE_BRIEF["height"],
            "width": observed["width"] == COURSE_BRIEF["width"],
            "channels": observed["n_channels"] == COURSE_BRIEF["channels"],
            "alpha_zeta_combinations": observed["alpha_zeta_combinations"] == COURSE_BRIEF["alpha_zeta_combinations"],
        },
        "difference_note": (
            "Local copy is raw trajectory shards with 81x256x256 fields; "
            "the 16x224 processed windows described in the brief are not what is stored locally."
            if observed["layout"] == "raw_trajectory_fields"
            else "Local layout differs from the raw The Well layout; inspect audit outputs."
        ),
    }


def generate_report(
    out_dir: Path,
    layout_summary: Dict[str, object],
    analytics: Dict[str, pd.DataFrame],
    baseline_summary: Optional[pd.DataFrame],
    brief_comparison: Dict[str, object],
) -> None:
    """Generate the markdown report with actionable SSL conclusions."""
    channel_stats = analytics["channel_stats"]
    smoothness = analytics["smoothness"]
    spectral = analytics["spectral"]
    sample_summary = analytics["sample_summary"]
    corr_matrix = analytics["channel_corr"]

    grouped_channel = channel_stats.groupby("channel", as_index=False).agg(
        mean=("mean", "mean"),
        std=("std", "mean"),
        min=("min", "min"),
        max=("max", "max"),
        nan_count=("nan_count", "sum"),
        inf_count=("inf_count", "sum"),
        is_constant=("is_constant", "mean"),
    )
    robust_recommendation = "robust scaling"
    if float(grouped_channel["nan_count"].sum()) == 0 and float(grouped_channel["inf_count"].sum()) == 0:
        if float((grouped_channel["max"] - grouped_channel["min"]).max()) < 25.0:
            robust_recommendation = "per-channel z-score"

    easy_score = float(sample_summary["adjacent_prediction_easy_score"].mean())
    decorrelation = float(sample_summary["decorrelation_time_mean"].mean())
    stride_useful = "likely yes" if decorrelation > 2.0 or easy_score > 0.7 else "unclear"
    context_enough = "probably yes for short-horizon SSL pretraining, but test longer horizons in ablations" if decorrelation <= 16 else "possibly short; compare 16 vs 24/32"

    corr_pairs = (
        corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "channel_a", "level_1": "channel_b", 0: "corr"})
    )
    corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
    corr_pairs = corr_pairs.sort_values("abs_corr", ascending=False)
    redundant = corr_pairs.drop_duplicates(subset=["abs_corr"]).head(5)

    regime_df = sample_summary.groupby(["alpha", "zeta"], as_index=False).mean(numeric_only=True)
    easiest = regime_df.sort_values("adjacent_prediction_easy_score", ascending=False).head(5)
    hardest = regime_df.sort_values("adjacent_prediction_easy_score", ascending=True).head(5)

    lines: List[str] = []
    lines.append("# Active Matter Dataset Report")
    lines.append("")
    lines.append("## Actual Local Structure")
    lines.append(f"- Layout detected: `{brief_comparison['observed_local_copy']['layout']}`")
    lines.append(f"- Files scanned: {layout_summary['num_files']}")
    lines.append(f"- Split file counts: {layout_summary['split_file_counts']}")
    lines.append(f"- Unique alpha values: {layout_summary['unique_alpha_values']}")
    lines.append(f"- Unique zeta values: {layout_summary['unique_zeta_values']}")
    lines.append(f"- Unique alpha-zeta combinations: {layout_summary['num_alpha_zeta_combinations']}")
    lines.append(f"- Total declared logical samples: {layout_summary['total_declared_samples']}")
    lines.append(f"- Verification against course brief: {brief_comparison['difference_note']}")
    lines.append("")
    lines.append("## Channel Interpretation")
    lines.append("- Local files expose one scalar field `concentration`, one vector field `velocity`, and two 2x2 tensor fields `D` and `E`.")
    lines.append("- The inferred 11 channels are concentration, 2 velocity components, 4 `D` components, and 4 `E` components.")
    lines.append("- If you need semantic names beyond `D` and `E`, keep them marked uncertain unless confirmed by upstream Well metadata or the generating code.")
    lines.append("")
    lines.append("## Integrity Checks")
    lines.append(f"- Total NaN count across analyzed samples: {int(channel_stats['nan_count'].sum())}")
    lines.append(f"- Total inf count across analyzed samples: {int(channel_stats['inf_count'].sum())}")
    lines.append(f"- Constant-channel suspicion rate: {grouped_channel['is_constant'].mean():.3f}")
    lines.append("- Use the saved CSVs for exact per-channel percentiles and split-level balance checks.")
    lines.append("")
    lines.append("## Normalization Recommendation")
    lines.append(f"- Recommended default: **{robust_recommendation}**.")
    lines.append("- Keep normalization per channel, not per full sample, so alpha/zeta-dependent amplitude differences are not erased unpredictably.")
    lines.append("- Also compare robust scaling on heavy-tailed tensor channels if histogram plots show extreme outliers.")
    lines.append("")
    lines.append("## Temporal Conclusions")
    lines.append(f"- Mean adjacent-frame easy score: {easy_score:.3f}")
    lines.append(f"- Mean decorrelation lag: {decorrelation:.2f} frames")
    lines.append(f"- Is temporal stride > 1 likely useful? **{stride_useful}**")
    lines.append(f"- Is 16-frame context enough? **{context_enough}**")
    lines.append("- If adjacent prediction looks too easy, start with stride 2 or 4 and add harder forecasting or masked modeling objectives.")
    lines.append("")
    lines.append("## Spatial Structure")
    lines.append(f"- Mean low/high frequency ratio: {sample_summary['spectral_low_high_ratio_mean'].mean():.3f}")
    lines.append(f"- Mean anisotropy ratio: {sample_summary['spectral_anisotropy_mean'].mean():.3f}")
    lines.append(f"- Mean spatial correlation length: {sample_summary['spatial_correlation_length_mean'].mean():.3f}")
    lines.append("- Use the FFT and regime heatmaps to spot parameter regions with sharper scale separation or stronger anisotropy.")
    lines.append("")
    lines.append("## Redundancy And Shortcuts")
    lines.append("- Most correlated channel pairs in sampled data:")
    for _, row in redundant.iterrows():
        lines.append(f"  - `{row['channel_a']}` vs `{row['channel_b']}`: corr={row['corr']:.3f}")
    lines.append("- Shortcut risks include parameter-specific amplitude fingerprints, repeated trajectories from the same alpha-zeta shard, and very easy adjacent-frame prediction.")
    lines.append("- Favor file-grouped splits for probes and keep same-file windows out of both train and eval in SSL diagnostics.")
    lines.append("")
    lines.append("## Physically Safe vs Unsafe Augmentations")
    lines.append("- Safer: random spatial crops, flips/rotations that preserve tensor/vector bookkeeping, mild Gaussian noise, time masking, channel dropout within redundant tensor groups.")
    lines.append("- Riskier: arbitrary channel mixing, sign flips on vector/tensor channels without symmetry checks, strong blurs that erase spectra, temporal reversal unless the physics objective tolerates it.")
    lines.append("")
    lines.append("## SSL Objectives To Prioritize")
    lines.append("- Start with masked spatiotemporal reconstruction or masked autoencoding over channels and patches.")
    lines.append("- Compare contrastive or VICReg-style objectives on trajectory windows only after checking duplicate-risk tables.")
    lines.append("- Add temporal-order or stride-aware forecasting only if the autocorrelation plots suggest adjacent prediction is too trivial.")
    lines.append("")
    lines.append("## First Ablations")
    lines.append("- Per-channel z-score vs robust scaling")
    lines.append("- Temporal stride 1 vs 2 vs 4")
    lines.append("- Full 11 channels vs dropping the most redundant tensor components")
    lines.append("- 16-frame windows vs longer context if decorrelation extends well beyond 16")
    lines.append("- Same-trajectory negative mining disabled vs enabled for contrastive methods")
    lines.append("")
    lines.append("## Cheap Baselines")
    if baseline_summary is None or baseline_summary.empty:
        lines.append("- Baseline regression results were not available when the report was generated.")
    else:
        for _, row in baseline_summary.iterrows():
            lines.append(
                f"- `{row['target']}` with `{row['model']}`: RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, R2={row['r2']:.3f}"
            )
        lines.append("- If simple summary features already predict alpha or zeta well, expect SSL probes to overfit easy global statistics unless augmentations remove them.")
    lines.append("")
    lines.append("## Regime Difficulty Signals")
    lines.append("- Easiest parameter regions by adjacent-frame easy score:")
    for _, row in easiest.iterrows():
        lines.append(f"  - alpha={row['alpha']}, zeta={row['zeta']}, score={row['adjacent_prediction_easy_score']:.3f}")
    lines.append("- Hardest parameter regions by adjacent-frame easy score:")
    for _, row in hardest.iterrows():
        lines.append(f"  - alpha={row['alpha']}, zeta={row['zeta']}, score={row['adjacent_prediction_easy_score']:.3f}")
    lines.append("")
    lines.append("## Simple Baselines Before Expensive Training")
    lines.append("- Linear probe on frozen summary features")
    lines.append("- kNN regression on PCA-reduced handcrafted features")
    lines.append("- Ridge regression with grouped CV by source file")
    lines.append("- Channel ablations using only concentration+velocity or only tensor channels")
    lines.append("- Forecasting baselines with mean-last-frame or low-rank PCA reconstructions")

    (out_dir / "active_matter_dataset_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    plot_dir = ensure_dir(out_dir / "plots")
    table_dir = ensure_dir(out_dir / "tables")
    report_dir = out_dir

    file_limit = args.max_files if args.max_files is not None else (12 if args.quick_mode else None)
    sample_limit = args.max_samples if args.max_samples is not None else (2 if args.quick_mode else None)

    files = discover_data_files(args.data_root)
    if file_limit is not None:
        files = balanced_file_subset(files, args.data_root, file_limit)
    if not files:
        raise SystemExit(f"No data files found under {args.data_root}")
    metadata_map = {str(path.resolve()): inspect_data_file(path, data_root=args.data_root) for path in files}
    layout_summary = summarize_layouts(list(metadata_map.values()))
    save_json(out_dir / "layout_summary.json", layout_summary)

    analytics = collect_sample_analytics(
        files=files,
        metadata_map=metadata_map,
        max_samples_per_file=sample_limit,
        sample_stride=args.sample_stride,
        out_dir=out_dir,
        skip_plots=args.skip_plots,
    )

    feature_df = build_feature_table(
        data_root=args.data_root,
        out_dir=out_dir,
        max_files=file_limit,
        max_samples=sample_limit,
        sample_stride=args.sample_stride,
        quick_mode=args.quick_mode,
    )
    feature_cols = [
        col for col in feature_df.columns
        if col not in {"sample_id", "file_path", "split", "alpha", "zeta", "sample_index"}
    ]
    duplicate_df = duplicate_suspicion_scores(feature_df, feature_cols)
    duplicate_df.to_csv(table_dir / "duplicate_suspicion.csv", index=False)
    embedding_df = pca_embedding(feature_df, feature_cols)
    embedding_df.to_csv(table_dir / "feature_pca_embedding.csv", index=False)

    if not args.skip_plots:
        plot_label_distributions(feature_df, plot_dir)
        plot_pca_embedding(embedding_df, plot_dir / "feature_pca.png")

    regime_df = feature_df.groupby(["alpha", "zeta"], as_index=False).mean(numeric_only=True)
    regime_df.to_csv(table_dir / "regime_summary.csv", index=False)
    if not args.skip_plots:
        plot_regime_heatmaps(
            regime_df,
            metrics=[
                "temporal_delta_energy_mean",
                "adjacent_prediction_easy_score",
                "spectral_low_high_ratio_mean",
                "spatial_correlation_length_mean",
            ],
            out_dir=plot_dir,
        )

    baseline_results, _ = run_regression_baselines(feature_df)
    baseline_results.to_csv(table_dir / "baseline_cv_results.csv", index=False)
    baseline_summary_df = baseline_results.groupby(["target", "model"], as_index=False).agg(
        {"rmse": "mean", "mae": "mean", "r2": "mean"}
    )
    baseline_summary_df.to_csv(table_dir / "baseline_summary.csv", index=False)

    brief_comparison = compare_to_brief(metadata_map)
    save_json(out_dir / "brief_comparison.json", brief_comparison)
    generate_report(out_dir, layout_summary, analytics, baseline_summary_df, brief_comparison)
    print(f"Saved analytics outputs to {out_dir}")


if __name__ == "__main__":
    main()
