"""Plotting helpers for dataset forensics outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_figure(fig: plt.Figure, path: Path, dpi: int = 160) -> None:
    """Tight-layout save wrapper."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_example_frames(
    array: np.ndarray,
    channel_names: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    """Grid of early/mid/late frames for each channel."""
    n_channels = len(channel_names)
    frame_ids = sorted({0, max(array.shape[0] // 2, 0), array.shape[0] - 1})
    fig, axes = plt.subplots(n_channels, len(frame_ids), figsize=(4 * len(frame_ids), 2.5 * n_channels))
    axes = np.atleast_2d(axes)
    for row, channel in enumerate(channel_names):
        for col, frame_idx in enumerate(frame_ids):
            ax = axes[row, col]
            im = ax.imshow(array[frame_idx, :, :, row], cmap="viridis")
            ax.set_title(f"{channel} | t={frame_idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    save_figure(fig, out_path)


def plot_channel_histograms(
    stats_df: pd.DataFrame,
    value_frame: pd.DataFrame,
    out_path: Path,
) -> None:
    """Histogram grid for channel values."""
    channels = list(value_frame.columns)
    n_cols = 3
    n_rows = int(np.ceil(len(channels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows))
    axes = axes.flatten()
    for idx, channel in enumerate(channels):
        ax = axes[idx]
        sns.histplot(value_frame[channel], bins=60, ax=ax, stat="density", element="step", fill=True)
        ax.set_title(channel)
    for ax in axes[len(channels):]:
        ax.axis("off")
    fig.suptitle("Per-channel value histograms")
    save_figure(fig, out_path)


def plot_label_distributions(feature_df: pd.DataFrame, out_dir: Path) -> None:
    """Alpha/zeta marginal distributions and joint heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=feature_df, x="alpha", ax=axes[0], color="#377eb8")
    axes[0].set_title("Alpha distribution")
    sns.countplot(data=feature_df, x="zeta", ax=axes[1], color="#4daf4a")
    axes[1].set_title("Zeta distribution")
    save_figure(fig, out_dir / "label_distributions.png")

    pivot = feature_df.pivot_table(index="zeta", columns="alpha", values="sample_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako", ax=ax)
    ax.set_title("Alpha-Zeta sample count heatmap")
    save_figure(fig, out_dir / "alpha_zeta_heatmap.png")


def plot_correlation_heatmap(corr: np.ndarray, channel_names: Sequence[str], out_path: Path) -> None:
    """Channel correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, xticklabels=channel_names, yticklabels=channel_names, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Channel correlation heatmap")
    save_figure(fig, out_path)


def plot_temporal_summary(smoothness_df: pd.DataFrame, autocorr_df: pd.DataFrame, out_dir: Path) -> None:
    """Temporal dynamics plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=smoothness_df, x="channel", y="delta_to_signal_ratio", ax=axes[0], color="#984ea3")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Frame-delta / signal energy ratio")
    sns.lineplot(data=autocorr_df, x="lag", y="autocorr", hue="channel", ax=axes[1], legend=False)
    axes[1].set_title("Temporal autocorrelation")
    save_figure(fig, out_dir / "temporal_dynamics.png")


def plot_spectra(spectra: List[Dict[str, np.ndarray]], out_path: Path) -> None:
    """Radial power spectra."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for item in spectra:
        ax.plot(item["radius"], item["power"], label=item["channel"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Power")
    ax.set_title("Radial power spectra")
    ax.legend(ncol=2, fontsize=8)
    save_figure(fig, out_path)


def plot_regime_heatmaps(regime_df: pd.DataFrame, metrics: Sequence[str], out_dir: Path) -> None:
    """Heatmaps over alpha/zeta for selected regime summaries."""
    for metric in metrics:
        pivot = regime_df.pivot_table(index="zeta", columns="alpha", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, cmap="crest", annot=True, fmt=".2f", ax=ax)
        ax.set_title(metric.replace("_", " "))
        save_figure(fig, out_dir / f"regime_{metric}.png")


def plot_pca_embedding(embedding_df: pd.DataFrame, out_path: Path) -> None:
    """PCA scatter colored by zeta with alpha annotations available in hover-free form."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(embedding_df["pc1"], embedding_df["pc2"], c=embedding_df["zeta"], cmap="viridis", alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trajectory summary feature PCA")
    fig.colorbar(sc, ax=ax, label="zeta")
    save_figure(fig, out_path)


def plot_baseline_results(results_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart for baseline metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["rmse", "r2"]):
        sns.barplot(data=results_df, x="model", y=metric, hue="target", ax=ax)
        ax.set_title(f"Baseline {metric.upper()}")
        ax.tick_params(axis="x", rotation=20)
    save_figure(fig, out_path)
