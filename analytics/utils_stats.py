"""Statistics and feature helpers for active matter dataset forensics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import fft, spatial
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


def safe_corrcoef(matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix that tolerates constant columns."""
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    std = matrix.std(axis=0)
    active = std > 0
    corr = np.eye(matrix.shape[1], dtype=np.float64)
    if active.sum() >= 2:
        corr_active = np.corrcoef(matrix[:, active], rowvar=False)
        corr[np.ix_(active, active)] = corr_active
    return corr


def per_channel_stats(array: np.ndarray, channel_names: Sequence[str]) -> pd.DataFrame:
    """Compute global statistics per channel for one sample."""
    flat = array.reshape(-1, array.shape[-1])
    rows = []
    for idx, name in enumerate(channel_names):
        values = flat[:, idx]
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        if finite.size == 0:
            rows.append(
                {
                    "channel": name,
                    "count": values.size,
                    "finite_count": 0,
                    "nan_count": int(np.isnan(values).sum()),
                    "inf_count": int(np.isinf(values).sum()),
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "is_constant": True,
                }
            )
            continue
        row = {
            "channel": name,
            "count": int(values.size),
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "is_constant": bool(np.std(finite) < 1e-12),
        }
        for q in PERCENTILES:
            row[f"p{str(q).replace('.', '_')}"] = float(np.percentile(finite, q))
        rows.append(row)
    return pd.DataFrame(rows)


def sample_feature_summary(array: np.ndarray, channel_names: Sequence[str]) -> Dict[str, Any]:
    """Hand-crafted summary features for one sample/trajectory."""
    T, H, W, C = array.shape
    flat = array.reshape(T * H * W, C)
    features: Dict[str, Any] = {}
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    medians = np.median(flat, axis=0)
    q05 = np.percentile(flat, 5, axis=0)
    q95 = np.percentile(flat, 95, axis=0)
    for idx, name in enumerate(channel_names):
        features[f"{name}__mean"] = float(means[idx])
        features[f"{name}__std"] = float(stds[idx])
        features[f"{name}__median"] = float(medians[idx])
        features[f"{name}__q05"] = float(q05[idx])
        features[f"{name}__q95"] = float(q95[idx])

    temporal = temporal_metrics(array, channel_names)
    spectral = spectral_metrics(array, channel_names)
    corr = cross_channel_correlation(array, channel_names)
    for key, value in temporal["scalar_summary"].items():
        features[key] = value
    for key, value in spectral["scalar_summary"].items():
        features[key] = value
    for key, value in corr["scalar_summary"].items():
        features[key] = value
    return features


def cross_channel_correlation(array: np.ndarray, channel_names: Sequence[str]) -> Dict[str, Any]:
    """Cross-channel redundancy estimates."""
    flat = array.reshape(-1, array.shape[-1])
    if flat.shape[0] > 200000:
        indices = np.linspace(0, flat.shape[0] - 1, 200000, dtype=np.int64)
        flat = flat[indices]
    corr = safe_corrcoef(flat)
    rows = []
    for i, src in enumerate(channel_names):
        for j, dst in enumerate(channel_names):
            if j <= i:
                continue
            rows.append(
                {
                    "channel_a": src,
                    "channel_b": dst,
                    "corr": float(corr[i, j]),
                    "abs_corr": float(abs(corr[i, j])),
                }
            )
    pair_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    scalar_summary = {
        "channel_corr_abs_mean": float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)]))),
        "channel_corr_abs_max": float(np.max(np.abs(corr[np.triu_indices_from(corr, k=1)]))),
    }
    return {"matrix": corr, "pairs": pair_df, "scalar_summary": scalar_summary}


def _frame_energy_by_channel(array: np.ndarray) -> np.ndarray:
    """Average squared field magnitude per frame/channel."""
    return np.mean(array.astype(np.float64) ** 2, axis=(1, 2))


def _frame_autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Simple normalized autocorrelation for 1D series."""
    centered = series - np.mean(series)
    denom = np.dot(centered, centered)
    if denom <= 0:
        return np.ones(max_lag + 1, dtype=np.float64)
    out = [1.0]
    for lag in range(1, max_lag + 1):
        if lag >= centered.size:
            out.append(np.nan)
            continue
        out.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
    return np.asarray(out, dtype=np.float64)


def temporal_metrics(
    array: np.ndarray,
    channel_names: Sequence[str],
    max_lag: int = 12,
) -> Dict[str, Any]:
    """Temporal smoothness, frame deltas, and decorrelation estimates."""
    delta = np.diff(array.astype(np.float64), axis=0)
    delta_energy = np.mean(delta ** 2, axis=(1, 2))
    signal_energy = np.mean(array.astype(np.float64) ** 2, axis=(1, 2))
    smoothness = delta_energy.mean(axis=0) / np.maximum(signal_energy[:-1].mean(axis=0), 1e-12)

    autocorr_rows = []
    decorrelation = {}
    horizon_difficulty = {}
    for idx, name in enumerate(channel_names):
        series = signal_energy[:, idx]
        ac = _frame_autocorr(series, max_lag=max_lag)
        for lag, value in enumerate(ac):
            autocorr_rows.append({"channel": name, "lag": lag, "autocorr": float(value)})
        dec_lag = next((lag for lag, value in enumerate(ac[1:], start=1) if value < np.exp(-1)), np.nan)
        decorrelation[name] = float(dec_lag) if not np.isnan(dec_lag) else np.nan
        horizon_difficulty[name] = float(1.0 - np.nanmean(ac[1:min(4, len(ac))]))

    scalar_summary = {
        "temporal_delta_energy_mean": float(delta_energy.mean()),
        "temporal_delta_energy_std": float(delta_energy.std()),
        "adjacent_prediction_easy_score": float(1.0 - np.mean(smoothness)),
        "stride2_delta_energy_ratio": float(
            np.mean(np.diff(array.astype(np.float64), n=1, axis=0)[::2] ** 2) /
            max(np.mean(delta ** 2), 1e-12)
        ) if array.shape[0] >= 3 else np.nan,
        "decorrelation_time_mean": float(np.nanmean(list(decorrelation.values()))),
    }
    smoothness_df = pd.DataFrame(
        {
            "channel": channel_names,
            "delta_to_signal_ratio": smoothness,
            "decorrelation_lag": [decorrelation[name] for name in channel_names],
            "horizon_difficulty": [horizon_difficulty[name] for name in channel_names],
        }
    )
    return {
        "delta_energy": delta_energy,
        "autocorr": pd.DataFrame(autocorr_rows),
        "smoothness": smoothness_df,
        "scalar_summary": scalar_summary,
    }


def radial_power_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radially average a 2D power spectrum."""
    image = image.astype(np.float64)
    spec = np.abs(fft.fftshift(fft.fft2(image))) ** 2
    h, w = spec.shape
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
    radial_sum = np.bincount(r.ravel(), weights=spec.ravel())
    radial_count = np.bincount(r.ravel())
    radial_mean = radial_sum / np.maximum(radial_count, 1)
    radii = np.arange(radial_mean.size)
    return radii, radial_mean


def _spatial_autocorr_length(image: np.ndarray) -> float:
    """Estimate correlation length from FFT-based autocorrelation."""
    centered = image.astype(np.float64) - float(np.mean(image))
    corr = fft.ifft2(np.abs(fft.fft2(centered)) ** 2).real
    corr = fft.fftshift(corr)
    corr /= max(float(np.max(corr)), 1e-12)
    h, w = corr.shape
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
    radial_sum = np.bincount(r.ravel(), weights=corr.ravel())
    radial_count = np.bincount(r.ravel())
    radial = radial_sum / np.maximum(radial_count, 1)
    for radius, value in enumerate(radial[1:], start=1):
        if value < np.exp(-1):
            return float(radius)
    return float(len(radial) - 1)


def spectral_metrics(
    array: np.ndarray,
    channel_names: Sequence[str],
    frame_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Frequency-domain summaries from a few representative frames."""
    T = array.shape[0]
    if frame_indices is None:
        frame_indices = sorted({0, max(T // 2, 0), T - 1})
    spectra: List[Dict[str, Any]] = []
    scalar_summary: Dict[str, Any] = {}
    per_channel_rows = []

    for ch_idx, name in enumerate(channel_names):
        radial_stack = []
        low_high_ratios = []
        anisotropy_values = []
        corr_lengths = []
        for frame_idx in frame_indices:
            image = array[frame_idx, :, :, ch_idx]
            radii, radial = radial_power_spectrum(image)
            radial_stack.append(radial)
            cutoff = max(2, len(radial) // 8)
            low = float(np.sum(radial[:cutoff]))
            high = float(np.sum(radial[cutoff:]))
            low_high_ratios.append(float(np.log10((low + 1e-12) / (high + 1e-12))))

            spec = np.abs(fft.fftshift(fft.fft2(image.astype(np.float64)))) ** 2
            anisotropy = float(np.var(spec.mean(axis=0)) / max(np.var(spec.mean(axis=1)), 1e-12))
            anisotropy_values.append(anisotropy)
            corr_lengths.append(_spatial_autocorr_length(image))
        min_len = min(len(rad) for rad in radial_stack)
        mean_radial = np.mean([rad[:min_len] for rad in radial_stack], axis=0)
        spectra.append(
            {
                "channel": name,
                "radius": np.arange(min_len),
                "power": mean_radial,
            }
        )
        per_channel_rows.append(
            {
                "channel": name,
                "low_high_freq_ratio": float(np.mean(low_high_ratios)),
                "anisotropy_ratio": float(np.mean(anisotropy_values)),
                "spatial_correlation_length": float(np.mean(corr_lengths)),
            }
        )

    per_channel_df = pd.DataFrame(per_channel_rows)
    scalar_summary["spectral_low_high_ratio_mean"] = float(per_channel_df["low_high_freq_ratio"].mean())
    scalar_summary["spectral_anisotropy_mean"] = float(per_channel_df["anisotropy_ratio"].mean())
    scalar_summary["spatial_correlation_length_mean"] = float(per_channel_df["spatial_correlation_length"].mean())
    return {
        "spectra": spectra,
        "per_channel": per_channel_df,
        "scalar_summary": scalar_summary,
    }


def duplicate_suspicion_scores(feature_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """kNN-style duplicate suspicion based on standardized feature similarity."""
    X = feature_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    Xn = (X - mean) / np.maximum(std, 1e-12)
    distance = cdist(Xn, Xn, metric="euclidean")
    np.fill_diagonal(distance, np.inf)
    nearest_idx = np.argmin(distance, axis=1)
    nearest_dist = distance[np.arange(distance.shape[0]), nearest_idx]
    out = feature_df.loc[:, ["sample_id", "file_path", "alpha", "zeta", "split"]].copy()
    out["nearest_neighbor_index"] = nearest_idx
    out["nearest_neighbor_distance"] = nearest_dist
    out["duplicate_suspect"] = nearest_dist < np.percentile(nearest_dist, 10)
    return out.sort_values("nearest_neighbor_distance")


def pca_embedding(feature_df: pd.DataFrame, feature_cols: Sequence[str], n_components: int = 2) -> pd.DataFrame:
    """Low-dimensional embedding for summary features."""
    X = feature_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    scale = np.maximum(X.std(axis=0, keepdims=True), 1e-12)
    X = X / scale
    model = PCA(n_components=n_components, random_state=0)
    emb = model.fit_transform(X)
    out = feature_df.loc[:, ["sample_id", "alpha", "zeta", "split"]].copy()
    for idx in range(n_components):
        out[f"pc{idx + 1}"] = emb[:, idx]
    out["explained_variance_ratio"] = [model.explained_variance_ratio_.tolist()] * len(out)
    return out


def grouped_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    fold: int,
) -> Dict[str, Any]:
    """Consistent regression metrics for saved results."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "target": target_name,
        "model": model_name,
        "fold": fold,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
