"""
Training entry point for active-matter JEPA experiments.

The module intentionally keeps the dataset, model, evaluation, and training
code in one importable place so cluster jobs can run from a single package
entry point without relying on notebook state or scratch files.

Examples:

    python -m active_matter_jepa.train --config configs/vit_h_jepa_multiscale_a100.yaml
    am-jepa-train --config configs/local/h_jepa_multiscale_cpu_smoke.yaml --smoke_test
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML required: pip install pyyaml") from exc

try:
    import wandb
except ImportError:
    wandb = None


SHUTDOWN_REQUESTED = False


# ---------------------------------------------------------------------------
# Data and evaluation helpers (formerly imported from dataloader.py)
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    seed: int = 42
    data_root: str = "data"
    output_dir: str = "/scratch/${USER}/am_jepa/run"
    model_type: str = "global"
    file_patterns: Tuple[str, ...] = ("*.hdf5", "*.h5")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    context_chunk_len: int = 4
    target_chunk_len: int = 4
    gap: int = 16
    frame_stride: int = 4
    window_start_stride: int = 8
    max_windows_per_traj: Optional[int] = None

    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 30
    warmup_epochs: int = 2
    smoke_batches_per_epoch: int = 2
    lr: float = 3e-4
    weight_decay: float = 5e-2
    grad_clip_norm: float = 1.0
    lambda_inv: float = 25.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0
    vicreg_eps: float = 1e-4

    embed_dim: int = 384
    input_channels: int = 11
    latent_channels: int = 128
    spatial_eval_pool: int = 4
    knn_k: int = 10
    collapse_var_threshold: float = 1e-3
    collapse_rank_threshold: float = 5.0
    collapse_cos_threshold: float = 0.95

    stats_max_windows: int = 512
    save_every_steps: int = 500
    eval_every_epochs: int = 1
    log_every_steps: int = 25
    max_train_windows_per_split: Optional[int] = None
    max_eval_windows_per_split: Optional[int] = None
    smoke_max_windows_per_split: int = 16

    wandb_project: str = "active-matter-jepa"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    resume: bool = True

def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_rank0(rank: int) -> bool:
    return rank == 0

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model

def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

def atomic_json_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)

def safe_scalar(x: Any) -> float:
    if isinstance(x, h5py.Dataset):
        x = x[()]
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[0])

def all_dataset_paths(h5: h5py.File) -> List[str]:
    paths: List[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5.visititems(visitor)
    return sorted(paths)

def find_dataset_key(
    h5: h5py.File, candidates: Sequence[str], contains: Optional[Sequence[str]] = None
) -> str:
    for key in candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return key
    paths = all_dataset_paths(h5)
    if contains:
        tokens = [token.lower() for token in contains]
        basename_matches = [p for p in paths if Path(p).name.lower() in tokens]
        if basename_matches:
            return basename_matches[0]
        matches = [p for p in paths if all(token in p.lower() for token in tokens)]
        if matches:
            return matches[0]
    raise KeyError(
        f"Could not find dataset. Tried {list(candidates)}. Available: {paths[:40]}"
    )

def read_optional_scalar(h5: h5py.File, name: str) -> float:
    for key in (f"scalars/{name}", name):
        if key in h5:
            return safe_scalar(h5[key])
    if name in h5.attrs:
        return safe_scalar(h5.attrs[name])
    raise KeyError(f"Missing scalar {name}")

def _read_traj_dataset(
    h5: h5py.File, key: str, traj_idx: int, expected_tail_ndim: int
) -> np.ndarray:
    ds = h5[key]
    if ds.ndim == expected_tail_ndim:
        if traj_idx != 0:
            raise IndexError(
                f"Dataset {key} has no trajectory axis; traj_idx must be 0"
            )
        arr = ds[()]
    elif ds.ndim == expected_tail_ndim + 1:
        if traj_idx >= ds.shape[0]:
            raise IndexError(
                f"traj_idx={traj_idx} out of range for {key} with {ds.shape[0]} trajectories"
            )
        arr = ds[traj_idx]
    else:
        raise ValueError(f"Unexpected shape for {key}: {ds.shape}")
    return np.asarray(arr, dtype=np.float32)

def load_active_matter_trajectory(path: Path, traj_idx: int = 0) -> Dict[str, Any]:
    with h5py.File(path, "r") as h5:
        c_key = find_dataset_key(
            h5,
            ["t0_fields/concentration", "fields/concentration", "concentration"],
            ["concentration"],
        )
        v_key = find_dataset_key(
            h5, ["t1_fields/velocity", "fields/velocity", "velocity"], ["velocity"]
        )
        d_key = find_dataset_key(h5, ["t2_fields/D", "fields/D", "D"], ["d"])
        e_key = find_dataset_key(h5, ["t2_fields/E", "fields/E", "E"], ["e"])
        concentration = _read_traj_dataset(h5, c_key, traj_idx, expected_tail_ndim=3)
        velocity = _read_traj_dataset(h5, v_key, traj_idx, expected_tail_ndim=4)
        d_tensor = _read_traj_dataset(h5, d_key, traj_idx, expected_tail_ndim=5)
        e_tensor = _read_traj_dataset(h5, e_key, traj_idx, expected_tail_ndim=5)
        alpha = read_optional_scalar(h5, "alpha")
        zeta = read_optional_scalar(h5, "zeta")
    assert (
        concentration.shape[:3]
        == velocity.shape[:3]
        == d_tensor.shape[:3]
        == e_tensor.shape[:3]
    )
    assert velocity.shape[-1] == 2
    assert d_tensor.shape[-2:] == (2, 2) and e_tensor.shape[-2:] == (2, 2)
    return {
        "concentration": concentration,
        "velocity": velocity,
        "D": d_tensor,
        "E": e_tensor,
        "alpha": float(alpha),
        "zeta": float(zeta),
        "file_path": str(path),
        "traj_idx": int(traj_idx),
    }

def stack_11_channels(sample: Dict[str, Any]) -> np.ndarray:
    c = sample["concentration"]
    v = sample["velocity"]
    d = sample["D"]
    e = sample["E"]
    channels = [
        c[..., None],
        v[..., 0:1],
        v[..., 1:2],
        d[..., 0, 0, None],
        d[..., 0, 1, None],
        d[..., 1, 0, None],
        d[..., 1, 1, None],
        e[..., 0, 0, None],
        e[..., 0, 1, None],
        e[..., 1, 0, None],
        e[..., 1, 1, None],
    ]
    out = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)
    assert out.ndim == 4 and out.shape[-1] == 11, out.shape
    return out

def list_hdf5_files(data_root: Path, patterns: Sequence[str]) -> List[Path]:
    split_dirs = [
        p
        for p in [data_root / "train", data_root / "valid", data_root / "test"]
        if p.exists()
    ]
    search_dirs = split_dirs if split_dirs else [data_root]
    files: List[Path] = []
    for directory in search_dirs:
        for pattern in patterns:
            files.extend(sorted(directory.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {data_root.resolve()}")
    return sorted(set(files))

def infer_num_trajectories(path: Path) -> Tuple[int, int, int, int, float, float]:
    with h5py.File(path, "r") as h5:
        c_key = find_dataset_key(
            h5,
            ["t0_fields/concentration", "fields/concentration", "concentration"],
            ["concentration"],
        )
        shape = tuple(h5[c_key].shape)
        if len(shape) == 3:
            n_traj = 1
            steps, height, width = shape
        elif len(shape) == 4:
            n_traj = int(shape[0])
            steps, height, width = shape[-3:]
        else:
            raise ValueError(f"Unexpected concentration shape in {path}: {shape}")
        alpha = read_optional_scalar(h5, "alpha")
        zeta = read_optional_scalar(h5, "zeta")
    return n_traj, int(steps), int(height), int(width), float(alpha), float(zeta)

def scan_trajectories(
    data_root: Path, cfg: DataConfig, show_progress: bool = True
) -> pd.DataFrame:
    rows = []
    for path in tqdm(
        list_hdf5_files(data_root, cfg.file_patterns),
        desc="Scanning trajectories",
        disable=not show_progress,
    ):
        n_traj, steps, height, width, alpha, zeta = infer_num_trajectories(path)
        for traj_idx in range(n_traj):
            rows.append(
                {
                    "file_path": str(path),
                    "source_split": path.parent.name,
                    "traj_idx": traj_idx,
                    "time_steps": steps,
                    "height": height,
                    "width": width,
                    "alpha": alpha,
                    "zeta": zeta,
                }
            )
    return pd.DataFrame(rows)

def assign_trajectory_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if set(out["source_split"].unique()).issuperset({"train", "valid", "test"}):
        out["split"] = out["source_split"].replace({"valid": "valid"})
        return out

    rng = np.random.default_rng(seed)
    files = np.array(sorted(out["file_path"].unique()))
    order = rng.permutation(len(files))
    n_train = int(round(0.7 * len(files)))
    n_valid = int(round(0.15 * len(files)))

    split_map = {}
    for file_path in files[order[:n_train]]:
        split_map[file_path] = "train"
    for file_path in files[order[n_train : n_train + n_valid]]:
        split_map[file_path] = "valid"
    for file_path in files[order[n_train + n_valid :]]:
        split_map[file_path] = "test"

    out["split"] = out["file_path"].map(split_map)
    return out

def build_window_starts(total_steps: int, cfg: DataConfig) -> List[int]:
    last_target_offset = (
        (cfg.context_chunk_len - 1) * cfg.frame_stride
        + cfg.gap
        + (cfg.target_chunk_len - 1) * cfg.frame_stride
    )
    total_required = last_target_offset + 1
    if total_steps < total_required:
        return []
    max_start = total_steps - total_required
    return list(range(0, max_start + 1, cfg.window_start_stride))

def context_target_indices(
    start: int, cfg: DataConfig
) -> Tuple[np.ndarray, np.ndarray]:
    context = start + np.arange(cfg.context_chunk_len) * cfg.frame_stride
    target_start = start + (cfg.context_chunk_len - 1) * cfg.frame_stride + cfg.gap
    target = target_start + np.arange(cfg.target_chunk_len) * cfg.frame_stride
    return context.astype(np.int64), target.astype(np.int64)

def build_window_index(trajectory_df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    rows = []
    for traj_counter, row in enumerate(trajectory_df.itertuples(index=False)):
        starts = build_window_starts(int(row.time_steps), cfg)
        if (
            cfg.max_windows_per_traj is not None
            and len(starts) > cfg.max_windows_per_traj
        ):
            rng = np.random.default_rng(cfg.seed + traj_counter)
            starts = sorted(
                rng.choice(
                    starts, size=cfg.max_windows_per_traj, replace=False
                ).tolist()
            )
        for start in starts:
            context_idx, target_idx = context_target_indices(int(start), cfg)
            rows.append(
                {
                    "file_path": row.file_path,
                    "traj_idx": int(row.traj_idx),
                    "split": row.split,
                    "start": int(start),
                    "context_indices": ",".join(map(str, context_idx.tolist())),
                    "target_indices": ",".join(map(str, target_idx.tolist())),
                    "alpha": float(row.alpha),
                    "zeta": float(row.zeta),
                    "time_steps": int(row.time_steps),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Window index is empty. Check chunk lengths and data paths.")
    return out

def compute_channel_stats(
    window_df: pd.DataFrame,
    cfg: DataConfig,
    out_path: Path,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    if out_path.exists():
        payload = np.load(out_path)
        return {
            "mean": payload["mean"].astype(np.float32),
            "std": payload["std"].astype(np.float32),
        }
    train_windows = window_df[window_df["split"] == "train"].reset_index(drop=True)
    if len(train_windows) == 0:
        raise RuntimeError("No train windows available for normalization stats")
    sample_indices = np.linspace(
        0,
        len(train_windows) - 1,
        min(cfg.stats_max_windows, len(train_windows)),
        dtype=int,
    )
    sum_c = np.zeros(cfg.input_channels, dtype=np.float64)
    sumsq_c = np.zeros(cfg.input_channels, dtype=np.float64)
    count = 0
    for idx in tqdm(
        sample_indices, desc="Computing channel stats", disable=not show_progress
    ):
        rec = train_windows.iloc[int(idx)]
        sample = load_active_matter_trajectory(Path(rec.file_path), int(rec.traj_idx))
        arr = stack_11_channels(sample)
        context_idx, target_idx = context_target_indices(int(rec.start), cfg)
        for clip in [arr[context_idx], arr[target_idx]]:
            flat = clip.reshape(-1, cfg.input_channels).astype(np.float64)
            sum_c += flat.sum(axis=0)
            sumsq_c += np.square(flat).sum(axis=0)
            count += flat.shape[0]
        del sample, arr
        gc.collect()
    mean = sum_c / count
    var = sumsq_c / count - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def correlation_length_e_delta(e_frame: np.ndarray) -> float:
    q = np.asarray(e_frame[..., 0, 0] - e_frame[..., 1, 1], dtype=np.float32)
    q = q - float(q.mean())
    denom = float(np.square(q).sum())
    if denom <= 1e-12:
        return 0.0
    power = np.fft.fft2(q)
    corr = np.fft.ifft2(power * np.conj(power)).real / denom
    corr = np.fft.fftshift(corr)
    h, w = corr.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(np.int32)
    max_r = min(h, w) // 2
    radial = np.bincount(r.ravel(), weights=corr.ravel(), minlength=max_r + 1)[
        : max_r + 1
    ]
    counts = np.bincount(r.ravel(), minlength=max_r + 1)[: max_r + 1]
    radial = radial / np.maximum(counts, 1)
    positive = np.where(radial[1:] > math.exp(-1))[0]
    if len(positive) == 0:
        return 1.0
    last = int(positive[-1]) + 1
    if (
        last + 1 < len(radial)
        and radial[last] >= math.exp(-1)
        and radial[last + 1] < math.exp(-1)
    ):
        y0, y1 = radial[last], radial[last + 1]
        frac = (math.exp(-1) - y0) / max(y1 - y0, -1e-12)
        return float(last + np.clip(frac, 0.0, 1.0))
    return float(last)

def nematic_order_s(e_frame: np.ndarray) -> float:
    q = np.asarray(e_frame, dtype=np.float64).mean(axis=(0, 1))
    q = 0.5 * (q + q.T)
    eigvals = np.linalg.eigvalsh(q)
    return float(eigvals[-1] - 0.5)

class ActiveMatterWindowDataset(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        cfg: DataConfig,
        stats: Dict[str, np.ndarray],
        max_windows: Optional[int] = None,
    ):
        self.windows = windows.reset_index(drop=True)
        if max_windows is not None and len(self.windows) > max_windows:
            self.windows = self.windows.iloc[:max_windows].reset_index(drop=True)
        self.cfg = cfg
        self.mean = stats["mean"].reshape(1, 1, 1, cfg.input_channels)
        self.std = stats["std"].reshape(1, 1, 1, cfg.input_channels)
        self._cache_key: Optional[Tuple[str, int]] = None
        self._cache_sample: Optional[Dict[str, Any]] = None
        self._cache_array: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.windows)

    def _get_sample_array(
        self, file_path: str, traj_idx: int
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        key = (file_path, traj_idx)
        if (
            self._cache_key == key
            and self._cache_sample is not None
            and self._cache_array is not None
        ):
            return self._cache_sample, self._cache_array
        sample = load_active_matter_trajectory(Path(file_path), traj_idx)
        arr = stack_11_channels(sample)
        self._cache_key = key
        self._cache_sample = sample
        self._cache_array = arr
        return sample, arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.windows.iloc[idx]
        sample, arr = self._get_sample_array(rec.file_path, int(rec.traj_idx))
        context_idx, target_idx = context_target_indices(int(rec.start), self.cfg)
        context = (arr[context_idx] - self.mean) / self.std
        target = (arr[target_idx] - self.mean) / self.std
        center_frame = int(context_idx[len(context_idx) // 2])
        e_frame = sample["E"][center_frame]
        return {
            "context": torch.from_numpy(context.astype(np.float32, copy=False)),
            "target": torch.from_numpy(target.astype(np.float32, copy=False)),
            "alpha": torch.tensor(float(rec.alpha), dtype=torch.float32),
            "zeta": torch.tensor(float(rec.zeta), dtype=torch.float32),
            "corr_length": torch.tensor(
                correlation_length_e_delta(e_frame), dtype=torch.float32
            ),
            "nematic_s": torch.tensor(nematic_order_s(e_frame), dtype=torch.float32),
        }

def make_loader(
    dataset: Dataset,
    cfg: DataConfig,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameter_counts(model: nn.Module) -> Dict[str, int]:
    raw_model = unwrap_model(model)
    encoder = count_parameters(raw_model.encoder) if hasattr(raw_model, "encoder") else 0
    predictor = (
        count_parameters(raw_model.predictor) if hasattr(raw_model, "predictor") else 0
    )
    total = count_parameters(raw_model)
    return {"encoder": encoder, "predictor": predictor, "total": total}

def flatten_repr(z: torch.Tensor) -> torch.Tensor:
    if z.ndim > 2:
        return z.flatten(1)
    return z

def vicreg_rows(z: torch.Tensor) -> torch.Tensor:
    """Return a [N, D] matrix in the exact representation space trained by VICReg.

    Global JEPA uses one row per sample: (B, C) -> (B, C).
    Spatial JEPA / ViTSpatial JEPA use one row per spatial location:
    (B, C, H, W) -> (B * H * W, C).
    """
    if z.ndim == 2:
        return z
    if z.ndim < 2:
        raise ValueError(f"Expected representation with channel dim, got {tuple(z.shape)}")
    channels = z.shape[1]
    return z.movedim(1, -1).reshape(-1, channels)

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def collapse_stats(z_rows: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    with torch.no_grad():
        z = z_rows.detach().float()
        if z.ndim != 2:
            raise ValueError(f"collapse_stats expects [N, D], got {tuple(z.shape)}")
        if z.shape[0] < 2:
            return {
                "var_mean": 0.0,
                "var_min": 0.0,
                "var_max": 0.0,
                "mean_offdiag_cos": 0.0,
                "offdiag_cov_mean_abs": 0.0,
                "effective_rank": 1.0,
            }
        per_dim_var = z.var(dim=0, unbiased=False)
        var_mean = per_dim_var.mean().item()
        var_min = per_dim_var.min().item()
        var_max = per_dim_var.max().item()
        z_n = F.normalize(z, dim=1)
        sim = z_n @ z_n.T
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        mean_offdiag_cos = sim[~eye].mean().item()
        zc = z - z.mean(dim=0, keepdim=True)
        if zc.shape[1] <= 8192:
            cov = (zc.T @ zc) / (zc.shape[0] - 1)
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
        else:
            gram = (zc @ zc.T) / (zc.shape[0] - 1)
            eigvals = torch.linalg.eigvalsh(gram).clamp_min(eps)
            cov = None
        probs = eigvals / eigvals.sum()
        eff_rank = torch.exp(-(probs * torch.log(probs)).sum()).item()
        if cov is None:
            cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
        offdiag_cov_mean_abs = off_diagonal(cov).abs().mean().item()
    return {
        "var_mean": var_mean,
        "var_min": var_min,
        "var_max": var_max,
        "mean_offdiag_cos": mean_offdiag_cos,
        "offdiag_cov_mean_abs": offdiag_cov_mean_abs,
        "effective_rank": eff_rank,
    }

def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

@torch.no_grad()
def extract_representations(
    model: nn.Module, loader: DataLoader, cfg: DataConfig, split: str
) -> Dict[str, np.ndarray]:
    model = unwrap_model(model)
    model.eval()
    feats, alphas, zetas, corr_lengths, nematic_s = [], [], [], [], []
    for batch in tqdm(loader, desc=f"Extract {split} {cfg.model_type}", leave=False):
        batch = move_batch(batch, cfg.device)
        if cfg.model_type in {"spatial", "vit_spatial"}:
            z = model.encode_repr(batch["context"], pool=cfg.spatial_eval_pool)
        else:
            z = model.encode_repr(batch["context"])
        feats.append(z.detach().cpu().numpy().astype(np.float32))
        alphas.append(batch["alpha"].detach().cpu().numpy().astype(np.float32))
        zetas.append(batch["zeta"].detach().cpu().numpy().astype(np.float32))
        corr_lengths.append(
            batch["corr_length"].detach().cpu().numpy().astype(np.float32)
        )
        nematic_s.append(batch["nematic_s"].detach().cpu().numpy().astype(np.float32))
    return {
        "X": np.concatenate(feats, axis=0),
        "y_raw": np.stack([np.concatenate(alphas), np.concatenate(zetas)], axis=1),
        "corr_length": np.concatenate(corr_lengths),
        "nematic_s": np.concatenate(nematic_s),
    }


@torch.no_grad()
def extract_hjepa_representations(
    model: nn.Module, loader: DataLoader, cfg: DataConfig, split: str
) -> Dict[str, Any]:
    model = unwrap_model(model)
    model.eval()
    feats: Dict[str, List[np.ndarray]] = {k: [] for k in ["l1", "l2", "l3", "concat"]}
    alphas, zetas, corr_lengths, nematic_s = [], [], [], []
    for batch in tqdm(loader, desc=f"Extract {split} h_jepa", leave=False):
        batch = move_batch(batch, cfg.device)
        feature_dict = model.extract_features(batch["context"])
        for key in feats:
            feats[key].append(feature_dict[key].detach().cpu().numpy().astype(np.float32))
        alphas.append(batch["alpha"].detach().cpu().numpy().astype(np.float32))
        zetas.append(batch["zeta"].detach().cpu().numpy().astype(np.float32))
        corr_lengths.append(batch["corr_length"].detach().cpu().numpy().astype(np.float32))
        nematic_s.append(batch["nematic_s"].detach().cpu().numpy().astype(np.float32))
    return {
        "features": {k: np.concatenate(v, axis=0) for k, v in feats.items()},
        "y_raw": np.stack([np.concatenate(alphas), np.concatenate(zetas)], axis=1),
        "corr_length": np.concatenate(corr_lengths),
        "nematic_s": np.concatenate(nematic_s),
    }


def normalize_labels(
    train_payload: Dict[str, np.ndarray], payload: Dict[str, np.ndarray]
) -> np.ndarray:
    mean = train_payload["y_raw"].mean(axis=0, keepdims=True)
    std = train_payload["y_raw"].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    return (payload["y_raw"] - mean) / std

def fit_eval_linear(
    train_payload: Dict[str, np.ndarray], eval_payload: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, float]]:
    model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    model.fit(train_payload["X"], train_payload["y"])
    pred = model.predict(eval_payload["X"])
    mse = mean_squared_error(eval_payload["y"], pred, multioutput="raw_values")
    return pred, {"alpha_mse": float(mse[0]), "zeta_mse": float(mse[1])}

def fit_eval_knn(
    train_payload: Dict[str, np.ndarray], eval_payload: Dict[str, np.ndarray], k: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    k_eff = min(k, len(train_payload["X"]))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=k_eff, weights="distance")),
        ]
    )
    model.fit(train_payload["X"], train_payload["y"])
    pred = model.predict(eval_payload["X"])
    mse = mean_squared_error(eval_payload["y"], pred, multioutput="raw_values")
    return pred, {"alpha_mse": float(mse[0]), "zeta_mse": float(mse[1])}

def probe_skip_reason(
    train_payload: Dict[str, np.ndarray],
    eval_payload: Dict[str, np.ndarray],
    smoke_test: bool,
) -> Optional[str]:
    min_train = 64 if smoke_test else 32
    min_eval = 32 if smoke_test else 16
    reasons: List[str] = []
    if len(train_payload["X"]) < min_train:
        reasons.append(f"train_samples<{min_train}")
    if len(eval_payload["X"]) < min_eval:
        reasons.append(f"eval_samples<{min_eval}")
    train_std = train_payload["y_raw"].std(axis=0)
    eval_std = eval_payload["y_raw"].std(axis=0)
    if train_std.shape[0] != 2 or eval_std.shape[0] != 2:
        reasons.append("unexpected_target_shape")
    else:
        if np.any(train_std < 1e-8):
            reasons.append("train_targets_degenerate")
        if np.any(eval_std < 1e-8):
            reasons.append("eval_targets_degenerate")
    if not reasons:
        return None
    return ",".join(reasons)

def max_spearman_per_dim(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    df = pd.DataFrame(x)
    target = pd.Series(y, name="target")
    corr = (
        df.corrwith(target, method="spearman").abs().replace([np.inf, -np.inf], np.nan)
    )
    return float(corr.max(skipna=True))

def plot_zeta_scatter(
    y_true: np.ndarray, pred: np.ndarray, out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true[:, 1], pred[:, 1], s=18, alpha=0.7)
    low = min(float(y_true[:, 1].min()), float(pred[:, 1].min()))
    high = max(float(y_true[:, 1].max()), float(pred[:, 1].max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("true normalized zeta")
    ax.set_ylabel("predicted normalized zeta")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def run_frozen_eval(
    cfg: DataConfig,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    collapse_violated: bool,
    smoke_test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = unwrap_model(model)
    if cfg.model_type == "h_jepa_multiscale":
        reprs = {
            "train": extract_hjepa_representations(model, train_loader, cfg, "train"),
            "valid": extract_hjepa_representations(model, valid_loader, cfg, "valid"),
            "test": extract_hjepa_representations(model, test_loader, cfg, "test"),
        }
        for split in reprs:
            reprs[split]["y"] = normalize_labels(reprs["train"], reprs[split]).astype(np.float32)

        rows: List[Dict[str, Any]] = []
        feature_keys = ["l1", "l2", "l3", "concat"]
        for split in ["valid", "test"]:
            for feature_key in feature_keys:
                train_payload = {
                    "X": reprs["train"]["features"][feature_key],
                    "y": reprs["train"]["y"],
                    "y_raw": reprs["train"]["y_raw"],
                }
                eval_payload = {
                    "X": reprs[split]["features"][feature_key],
                    "y": reprs[split]["y"],
                    "y_raw": reprs[split]["y_raw"],
                }
                skip_reason = probe_skip_reason(train_payload, eval_payload, smoke_test)
                for probe_name in ["linear", f"knn-{cfg.knn_k}"]:
                    row = {
                        "model": cfg.model_type,
                        "split": split,
                        "feature": feature_key,
                        "probe": probe_name,
                        "collapse_violated": collapse_violated,
                        "skipped_due_to_small_or_degenerate_split": skip_reason is not None,
                        "skip_reason": skip_reason or "",
                        "alpha_mse": float("nan"),
                        "zeta_mse": float("nan"),
                        f"alpha_mse_{feature_key}": float("nan"),
                        f"zeta_mse_{feature_key}": float("nan"),
                    }
                    if skip_reason is None:
                        if probe_name == "linear":
                            _pred, metrics = fit_eval_linear(train_payload, eval_payload)
                        else:
                            _pred, metrics = fit_eval_knn(train_payload, eval_payload, cfg.knn_k)
                        row.update(metrics)
                        row[f"alpha_mse_{feature_key}"] = metrics["alpha_mse"]
                        row[f"zeta_mse_{feature_key}"] = metrics["zeta_mse"]
                    rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_dir / "frozen_probe_results.csv", index=False)
        atomic_json_save({"rows": rows}, output_dir / "frozen_probe_results.json")

        diag_rows = []
        for split, payload in reprs.items():
            for feature_key in feature_keys:
                x = payload["features"][feature_key]
                diag_rows.append({
                    "model": cfg.model_type,
                    "split": split,
                    "feature": feature_key,
                    "max_spearman_corr_length": max_spearman_per_dim(x, payload["corr_length"]),
                    "max_spearman_nematic_s": max_spearman_per_dim(x, payload["nematic_s"]),
                })
        diagnostics_df = pd.DataFrame(diag_rows)
        diagnostics_df.to_csv(output_dir / "representation_diagnostics.csv", index=False)
        atomic_json_save({"rows": diag_rows}, output_dir / "representation_diagnostics.json")

        if wandb is not None and wandb.run is not None:
            for row in rows:
                prefix = f"eval/{row['split']}/{row['probe']}/{row['feature']}"
                payload = {
                    f"{prefix}/skipped_due_to_small_or_degenerate_split": bool(row["skipped_due_to_small_or_degenerate_split"])
                }
                if not row["skipped_due_to_small_or_degenerate_split"]:
                    payload.update({
                        f"{prefix}/alpha_mse": row["alpha_mse"],
                        f"{prefix}/zeta_mse": row["zeta_mse"],
                    })
                wandb.log(payload)
            wandb.save(str(output_dir / "*.csv"))
            wandb.save(str(output_dir / "*.json"))
        return results_df, diagnostics_df

    for p in model.encoder.parameters():
        p.requires_grad_(False)
    reprs = {
        "train": extract_representations(model, train_loader, cfg, "train"),
        "valid": extract_representations(model, valid_loader, cfg, "valid"),
        "test": extract_representations(model, test_loader, cfg, "test"),
    }
    for split in reprs:
        reprs[split]["y"] = normalize_labels(reprs["train"], reprs[split]).astype(
            np.float32
        )
    rows = []
    predictions = {}
    for split in ["valid", "test"]:
        skip_reason = probe_skip_reason(reprs["train"], reprs[split], smoke_test)
        if skip_reason is not None:
            for probe in ["linear", f"knn-{cfg.knn_k}"]:
                rows.append(
                    {
                        "model": cfg.model_type,
                        "split": split,
                        "probe": probe,
                        "collapse_violated": collapse_violated,
                        "skipped_due_to_small_or_degenerate_split": True,
                        "skip_reason": skip_reason,
                        "alpha_mse": float("nan"),
                        "zeta_mse": float("nan"),
                    }
                )
            continue

        lin_pred, lin_metrics = fit_eval_linear(reprs["train"], reprs[split])
        knn_pred, knn_metrics = fit_eval_knn(reprs["train"], reprs[split], cfg.knn_k)
        predictions[(split, "linear")] = lin_pred
        predictions[(split, "knn")] = knn_pred
        rows.append(
            {
                "model": cfg.model_type,
                "split": split,
                "probe": "linear",
                "collapse_violated": collapse_violated,
                "skipped_due_to_small_or_degenerate_split": False,
                "skip_reason": "",
                **lin_metrics,
            }
        )
        rows.append(
            {
                "model": cfg.model_type,
                "split": split,
                "probe": f"knn-{cfg.knn_k}",
                "collapse_violated": collapse_violated,
                "skipped_due_to_small_or_degenerate_split": False,
                "skip_reason": "",
                **knn_metrics,
            }
        )
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "frozen_probe_results.csv", index=False)
    atomic_json_save({"rows": rows}, output_dir / "frozen_probe_results.json")
    if ("test", "linear") in predictions:
        plot_zeta_scatter(
            reprs["test"]["y"],
            predictions[("test", "linear")],
            output_dir / "figures" / "predicted_vs_true_zeta_linear.png",
            f"{cfg.model_type} linear probe",
        )

    diag_rows = []
    for split, payload in reprs.items():
        diag_rows.append(
            {
                "model": cfg.model_type,
                "split": split,
                "max_spearman_corr_length": max_spearman_per_dim(
                    payload["X"], payload["corr_length"]
                ),
                "max_spearman_nematic_s": max_spearman_per_dim(
                    payload["X"], payload["nematic_s"]
                ),
            }
        )
    diagnostics_df = pd.DataFrame(diag_rows)
    diagnostics_df.to_csv(output_dir / "representation_diagnostics.csv", index=False)
    atomic_json_save(
        {"rows": diag_rows}, output_dir / "representation_diagnostics.json"
    )

    if wandb is not None and wandb.run is not None:
        for row in rows:
            prefix = f"eval/{row['split']}/{row['probe']}"
            payload = {
                f"{prefix}/skipped_due_to_small_or_degenerate_split": bool(
                    row["skipped_due_to_small_or_degenerate_split"]
                )
            }
            if not row["skipped_due_to_small_or_degenerate_split"]:
                payload.update(
                    {
                        f"{prefix}/alpha_mse": row["alpha_mse"],
                        f"{prefix}/zeta_mse": row["zeta_mse"],
                    }
                )
            wandb.log(payload)
        for row in diag_rows:
            wandb.log(
                {
                    f"diagnostics/{row['split']}/max_spearman_corr_length": row[
                        "max_spearman_corr_length"
                    ],
                    f"diagnostics/{row['split']}/max_spearman_nematic_s": row[
                        "max_spearman_nematic_s"
                    ],
                }
            )
        wandb.save(str(output_dir / "*.csv"))
        wandb.save(str(output_dir / "*.json"))
        wandb.save(str(output_dir / "figures" / "*.png"))
    return results_df, diagnostics_df


# ---------------------------------------------------------------------------
# PhysicsViTJEPA model (formerly model.py)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# 1. GroupedPhysicsInputStem
# ---------------------------------------------------------------------------

class GroupedPhysicsInputStem(nn.Module):
    """
    Physics-aware channel grouping before patch embedding.

    Input layout of the 11 channels (from stack_11_channels):
        idx 0   : concentration  (scalar)
        idx 1-2 : velocity vx, vy  (vector)
        idx 3-6 : D tensor: D00, D01, D10, D11  (symmetric → D10==D01)
        idx 7-10: E tensor: E00, E01, E10, E11  (traceless sym → E10==E01, E11==-E00)

    Independent channels used:
        concentration : [0]          → 1 ch
        velocity      : [1, 2]       → 2 ch
        D (sym 2×2)   : [3, 4, 6]   → D00, D01, D11 (drop D10)
        E (traceless) : [7, 8]       → E00, E01 (drop E10, E11)

    Total independent: 1 + 2 + 3 + 2 = 8 channels (vs 11 raw).

    Each group is projected to embed_dim // 4 via a per-patch linear (Conv2d
    kernel=patch_size, stride=patch_size).  The four embeddings are summed.
    """

    # channel indices for each group (into the 11-channel stack)
    _IDX_C = [0]
    _IDX_V = [1, 2]
    _IDX_D = [3, 4, 6]   # D00, D01, D11
    _IDX_E = [7, 8]      # E00, E01

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 16,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        g = embed_dim // 4  # per-group dimension

        def _proj(in_ch: int) -> nn.Conv2d:
            return nn.Conv2d(in_ch, g, kernel_size=patch_size, stride=patch_size,
                             bias=True)

        self.proj_c = _proj(len(self._IDX_C))
        self.proj_v = _proj(len(self._IDX_V))
        self.proj_d = _proj(len(self._IDX_D))
        self.proj_e = _proj(len(self._IDX_E))

        # Merge projection: 4 × g → embed_dim  (pointwise, after summation)
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm  = nn.LayerNorm(embed_dim)

        n_patches = (img_size // patch_size) ** 2
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (B, C=11, H, W) — single normalised frame.
        Returns:
            tokens: (B, N_patches, embed_dim)
        """
        xc = frame[:, self._IDX_C]
        xv = frame[:, self._IDX_V]
        xd = frame[:, self._IDX_D]
        xe = frame[:, self._IDX_E]

        # Each proj: (B, g, H_p, W_p)
        ec = self.proj_c(xc)
        ev = self.proj_v(xv)
        ed = self.proj_d(xd)
        ee = self.proj_e(xe)

        # Concatenate along channel dim → (B, 4g, H_p, W_p)
        combined = torch.cat([ec, ev, ed, ee], dim=1)

        # Flatten spatial → tokens: (B, N, 4g)
        B, C4g, H_p, W_p = combined.shape
        tokens = combined.flatten(2).transpose(1, 2)  # (B, N, 4g)

        # Merge + norm
        tokens = self.norm(self.merge(tokens))  # (B, N, embed_dim)
        return tokens


# ---------------------------------------------------------------------------
# 2. ViT building blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # Stochastic depth (drop_path == 0 → identity)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Per-sample stochastic depth regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep) / keep
        return x * mask


# ---------------------------------------------------------------------------
# 3. VideoViTEncoder — per-frame 2D ViT
# ---------------------------------------------------------------------------

class VideoViTEncoder(nn.Module):
    """
    Processes T frames independently with a shared 2-D ViT, then mean-pools
    temporally to produce a (B, N_patches, embed_dim) spatial feature map.

    Key design choice: per-frame 2D rather than 3D tubelets.
    Rationale: preserves the spatial token structure required for ζ recovery
    and avoids initialization collapse seen with 3D convolutions on this
    smooth, slowly-evolving dataset.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        img_size: int = 224,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        self.img_size   = img_size
        n_h = img_size // patch_size
        n_w = img_size // patch_size
        self.n_patches  = n_h * n_w          # 196 at default settings
        self.grid_h     = n_h
        self.grid_w     = n_w

        # Physics-aware input stem (replaces naive flat Conv2d)
        self.stem = GroupedPhysicsInputStem(embed_dim, patch_size, img_size)

        # Learnable 2-D sinusoidal-initialised positional embedding
        self.pos_embed = nn.Parameter(
            self._build_2d_sincos_pos_embed(embed_dim, n_h, n_w),
            requires_grad=True,   # allow fine-tuning
        )

        # Stochastic depth schedule (linearly from 0 → drop_path_rate)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    @staticmethod
    def _build_2d_sincos_pos_embed(
        embed_dim: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Returns (1, grid_h*grid_w, embed_dim) sin-cos positional embedding."""
        assert embed_dim % 4 == 0
        half = embed_dim // 2
        quarter = embed_dim // 4

        gh = torch.arange(grid_h, dtype=torch.float32)
        gw = torch.arange(grid_w, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (torch.arange(quarter, dtype=torch.float32) / quarter))

        # Height embedding
        h_emb = gh.unsqueeze(1) * omega.unsqueeze(0)   # (grid_h, quarter)
        h_emb = torch.cat([h_emb.sin(), h_emb.cos()], dim=1)  # (grid_h, half)

        # Width embedding
        w_emb = gw.unsqueeze(1) * omega.unsqueeze(0)   # (grid_w, quarter)
        w_emb = torch.cat([w_emb.sin(), w_emb.cos()], dim=1)  # (grid_w, half)

        # Combine: (grid_h, grid_w, embed_dim)
        pos = torch.cat([
            h_emb.unsqueeze(1).expand(-1, grid_w, -1),
            w_emb.unsqueeze(0).expand(grid_h, -1, -1),
        ], dim=-1)
        return pos.reshape(1, grid_h * grid_w, embed_dim)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (B, 11, H, W) — single normalised physics frame.
        Returns:
            tokens: (B, N_patches, embed_dim) after transformer blocks.
        """
        x = self.stem(frame) + self.pos_embed   # (B, N, D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (B, T, H, W, C=11)  — normalised clip in dataset layout.
        Returns:
            spatial_map: (B, embed_dim, H_p, W_p)
                         where H_p = W_p = img_size // patch_size.
        """
        B, T, H, W, C = clip.shape
        expected_h = self.grid_h * self.patch_size
        expected_w = self.grid_w * self.patch_size
        if H != expected_h or W != expected_w:
            raise ValueError(
                f"Input frames are {H}x{W}, but model config has img_size={self.img_size} "
                f"and patch_size={self.patch_size}, which expects {expected_h}x{expected_w}. "
                "Set img_size in the YAML to match the dataset frame size."
            )
        # Reshape to (B*T, C, H, W) for per-frame processing
        frames = clip.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

        tokens = self._encode_single_frame(frames)          # (B*T, N, D)
        tokens = tokens.reshape(B, T, self.n_patches, self.embed_dim)
        tokens = tokens.mean(dim=1)                         # (B, N, D)  temporal mean-pool

        # Reshape to spatial grid
        spatial = tokens.transpose(1, 2).reshape(
            B, self.embed_dim, self.grid_h, self.grid_w
        )
        return spatial.contiguous()                         # (B, D, H_p, W_p)


# ---------------------------------------------------------------------------
# 4. ConvPredictor
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """
    Simple residual block: depthwise → pointwise → GELU → pointwise.
    Operates on (B, C, H, W) spatial feature maps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw   = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False)
        self.pw1  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(channels)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pw1(self.norm(self.dw(x)))
        x = self.pw2(self.act(x))
        return x + residual


class ConvPredictor(nn.Module):
    """
    Predicts z_target spatial feature map from z_context spatial feature map.

    Architecture: 1×1 expand → N residual blocks → 1×1 project back.
    width_mult=4 → hidden_dim = 4 × embed_dim.
    n_blocks=2 is the default (ablations in the reference paper show
    diminishing returns beyond 2 blocks for this data scale).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        width_mult: int = 4,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        hidden = embed_dim * width_mult

        self.expand   = nn.Conv2d(embed_dim, hidden, kernel_size=1, bias=False)
        self.expand_norm = LayerNorm2d(hidden)
        self.blocks   = nn.Sequential(*[ResidualConvBlock(hidden) for _ in range(n_blocks)])
        self.project  = nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim, H_p, W_p)
        Returns:
            pred: (B, embed_dim, H_p, W_p)
        """
        x = self.expand_norm(self.expand(z))
        x = self.blocks(x)
        return self.project(x)


# ---------------------------------------------------------------------------
# 5. PhysicsViTJEPA — top-level model
# ---------------------------------------------------------------------------

class PhysicsViTJEPA(nn.Module):
    """
    Full spatial ViT-JEPA model for active matter physical simulations.

    Forward pass:
        context clip → VideoViTEncoder → z_context (B, D, H_p, W_p)
        z_context     → ConvPredictor  → pred      (B, D, H_p, W_p)
        target clip   → VideoViTEncoder (no_grad)  → z_target   (B, D, H_p, W_p)

    VICReg loss is then applied jointly on pred and z_target spatial tokens.

    Evaluation (frozen probe):
        encode_repr(clip, pool=4) → (B, D × pool × pool) flat vector
        This is passed to a frozen linear or kNN regressor to predict α, ζ.
    """

    model_name = "physics_vit_jepa"

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        img_size: int = 224,
        drop_path_rate: float = 0.1,
        predictor_width_mult: int = 4,
        predictor_n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = VideoViTEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        self.predictor = ConvPredictor(
            embed_dim=embed_dim,
            width_mult=predictor_width_mult,
            n_blocks=predictor_n_blocks,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_map(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Returns the spatial feature map (B, D, H_p, W_p).
        clip: (B, T, H, W, C=11) or (B, T, H, W, 11)
        """
        return self.encoder(clip)

    def encode_repr(self, clip: torch.Tensor, pool: int = 4) -> torch.Tensor:
        """
        Flattened representation for downstream probing.
        Returns (B, D × pool × pool).
        """
        z = self.encode_map(clip)                                    # (B, D, H_p, W_p)
        z = F.adaptive_avg_pool2d(z, (pool, pool))                   # (B, D, pool, pool)
        return z.flatten(1)                                           # (B, D*pool*pool)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (B, T, H, W, 11) — normalised context clip.
            target:  (B, T, H, W, 11) — normalised target clip.
        Returns:
            pred:      (B, D, H_p, W_p) — predicted target representation.
            z_target:  (B, D, H_p, W_p) — actual target representation (no grad).
            z_context: (B, D, H_p, W_p) — context representation (for diagnostics).
        """
        z_context = self.encode_map(context)           # (B, D, H_p, W_p)
        pred      = self.predictor(z_context)          # (B, D, H_p, W_p)

        with torch.no_grad():
            z_target = self.encode_map(target)         # (B, D, H_p, W_p)

        return pred, z_target, z_context


# ---------------------------------------------------------------------------
# 6. Parameter counting utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_report(model: nn.Module) -> dict:
    if hasattr(model, "parameter_report") and not isinstance(model, PhysicsViTJEPA):
        return model.parameter_report()
    stem      = count_parameters(model.encoder.stem)
    attn      = count_parameters(model.encoder.blocks)
    pos       = model.encoder.pos_embed.numel()
    encoder   = count_parameters(model.encoder)
    predictor = count_parameters(model.predictor)
    total     = count_parameters(model)
    return {
        "stem_params":      stem,
        "attn_params":      attn,
        "pos_embed_params": pos,
        "encoder_params":   encoder,
        "predictor_params": predictor,
        "total_params":     total,
    }


# ---------------------------------------------------------------------------
# Multi-scale hierarchical JEPA
# ---------------------------------------------------------------------------

class HJEPATokenEncoder(nn.Module):
    """Per-frame 2D ViT encoder that returns token sequences instead of maps."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
        img_size: int,
        drop_path_rate: float = 0.1,
        frame_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.frame_batch_size = frame_batch_size

        self.stem = GroupedPhysicsInputStem(embed_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(
            self._build_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w),
            requires_grad=True,
        )
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    @staticmethod
    def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
        assert embed_dim % 4 == 0
        half = embed_dim // 2
        quarter = embed_dim // 4
        gh = torch.arange(grid_h, dtype=torch.float32)
        gw = torch.arange(grid_w, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
        h_emb = gh.unsqueeze(1) * omega.unsqueeze(0)
        h_emb = torch.cat([h_emb.sin(), h_emb.cos()], dim=1)
        w_emb = gw.unsqueeze(1) * omega.unsqueeze(0)
        w_emb = torch.cat([w_emb.sin(), w_emb.cos()], dim=1)
        pos = torch.cat([
            h_emb.unsqueeze(1).expand(-1, grid_w, -1),
            w_emb.unsqueeze(0).expand(grid_h, -1, -1),
        ], dim=-1)
        return pos.reshape(1, grid_h * grid_w, embed_dim)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(frame)
        if tokens.shape[1:] != self.pos_embed.shape[1:]:
            raise ValueError(
                f"Patch/token mismatch: stem produced {tuple(tokens.shape[1:])}, "
                f"pos_embed is {tuple(self.pos_embed.shape[1:])}. Check img_size and patch_size."
            )
        x = tokens + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"Expected clip shape [B,T,H,W,C], got {tuple(clip.shape)}")
        B, T, H, W, C = clip.shape
        if C != 11:
            raise ValueError(f"Expected 11 active-matter channels, got C={C}")
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input frames are {H}x{W}, but this level has img_size={self.img_size}. "
                "Set h_jepa levels or top-level img_size to match the dataset."
            )
        frames = clip.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        if self.frame_batch_size is not None and frames.shape[0] > self.frame_batch_size:
            chunks = [
                self._encode_single_frame(frames[i:i + self.frame_batch_size])
                for i in range(0, frames.shape[0], self.frame_batch_size)
            ]
            tokens = torch.cat(chunks, dim=0)
        else:
            tokens = self._encode_single_frame(frames)
        tokens = tokens.reshape(B, T, self.n_patches, self.embed_dim).mean(dim=1)
        expected = (B, self.n_patches, self.embed_dim)
        if tokens.shape != expected:
            raise RuntimeError(f"Expected tokens {expected}, got {tuple(tokens.shape)}")
        return tokens.contiguous()


class TokenPredictorBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = embed_dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TokenPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            TokenPredictorBlock(embed_dim, hidden_mult) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TokenPredictor expects [B,N,D], got {tuple(x.shape)}")
        return self.out_norm(self.blocks(x))


class SigRegLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, rows: torch.Tensor, track_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        if rows.ndim != 2:
            raise ValueError(f"SigReg expects [N,D], got {tuple(rows.shape)}")
        nonfinite = int(not torch.isfinite(rows).all().item())
        z = torch.nan_to_num(rows.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        z = z - z.mean(dim=0, keepdim=True)

        def _compute() -> Tuple[torch.Tensor, torch.Tensor]:
            try:
                singular_values = torch.linalg.svdvals(z)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"SigReg SVD failed for rows with shape {tuple(rows.shape)}; "
                    f"finite={not bool(nonfinite)}"
                ) from exc
            singular_values = torch.nan_to_num(
                singular_values, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            s_norm = singular_values / (singular_values.sum() + self.eps)
            entropy = -(s_norm * torch.log(s_norm + self.eps)).sum()
            return -entropy, entropy

        if track_grad:
            loss, entropy = _compute()
        else:
            with torch.no_grad():
                loss, entropy = _compute()
            loss = rows.sum() * 0.0 + loss.detach()

        effective_rank = torch.exp(entropy.detach()).item()
        denom = max(1, min(rows.shape[0], rows.shape[1]))
        return loss, {
            "sig_entropy": float(entropy.detach().cpu()),
            "effective_rank": float(effective_rank),
            "rank_ratio": float(effective_rank / denom),
            "nonfinite": float(nonfinite),
        }


class VICRegSigRegLoss(nn.Module):
    def __init__(
        self,
        lambda_inv: float,
        lambda_var: float,
        lambda_cov: float,
        lambda_sig: float,
        sigreg_eps: float = 1e-8,
        vicreg_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_sig = lambda_sig
        self.vicreg_eps = vicreg_eps
        self.sigreg = SigRegLoss(sigreg_eps)

    @staticmethod
    def _rows(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor [B,N,D], got {tuple(x.shape)}")
        return x.reshape(-1, x.shape[-1]).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}")
        target = target.detach()
        pred_rows = self._rows(pred)
        target_rows = self._rows(target)
        joint = torch.cat([pred_rows, target_rows], dim=0)
        joint_for_stats = torch.nan_to_num(joint.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)

        p_n = F.normalize(pred_rows, dim=-1, eps=1e-12)
        t_n = F.normalize(target_rows, dim=-1, eps=1e-12)
        cosine = (p_n * t_n).sum(dim=-1)
        loss_inv = 2.0 - 2.0 * cosine.mean()

        std = torch.sqrt(joint.var(dim=0, unbiased=False) + self.vicreg_eps)
        loss_var = F.relu(1.0 - std).mean()

        centered = joint - joint.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        loss_cov = off_diagonal(cov).pow(2).sum() / joint.shape[1]

        loss_sig, sig_metrics = self.sigreg(joint, track_grad=(self.lambda_sig != 0.0))
        total = (
            self.lambda_inv * loss_inv
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
            + self.lambda_sig * loss_sig
        )

        stats = collapse_stats(joint_for_stats)
        pred_norm = pred_rows.norm(dim=-1)
        target_norm = target_rows.norm(dim=-1)
        norms = torch.cat([pred_norm, target_norm])
        metrics = {
            **stats,
            **sig_metrics,
            "cosine_align_mean": float(cosine.detach().mean().cpu()),
            "repr_norm_mean": float(norms.detach().mean().cpu()),
            "repr_norm_std": float(norms.detach().std(unbiased=False).cpu()),
            "loss_nonfinite": float(not torch.isfinite(total.detach()).item()),
        }
        terms = {
            "loss_inv": loss_inv,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "loss_sig": loss_sig,
            "loss_total": total,
        }
        return total, terms, metrics


class HJEPALevel(nn.Module):
    def __init__(
        self,
        name: str,
        level_cfg: Dict[str, Any],
        img_size: int,
        mlp_ratio: int,
        drop_path_rate: float,
        predictor_depth: int,
        predictor_hidden_mult: int,
        sigreg_eps: float,
        vicreg_eps: float,
        frame_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.weight = float(level_cfg.get("weight", 1.0))
        self.loss_type = str(level_cfg.get("loss_type", "vicreg"))
        self.embed_dim = int(level_cfg["embed_dim"])
        self.patch_size = int(level_cfg["patch_size"])
        self.online_encoder = HJEPATokenEncoder(
            embed_dim=self.embed_dim,
            depth=int(level_cfg["depth"]),
            num_heads=int(level_cfg["num_heads"]),
            mlp_ratio=mlp_ratio,
            patch_size=self.patch_size,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            frame_batch_size=frame_batch_size,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = TokenPredictor(
            embed_dim=self.embed_dim,
            hidden_mult=predictor_hidden_mult,
            depth=predictor_depth,
        )
        self.loss_fn = VICRegSigRegLoss(
            lambda_inv=float(level_cfg.get("lambda_inv", 5.0)),
            lambda_var=float(level_cfg.get("lambda_var", 0.0)),
            lambda_cov=float(level_cfg.get("lambda_cov", 0.0)),
            lambda_sig=float(level_cfg.get("lambda_sig", 0.0)),
            sigreg_eps=sigreg_eps,
            vicreg_eps=vicreg_eps,
        )

    @property
    def grid_h(self) -> int:
        return self.online_encoder.grid_h

    @property
    def grid_w(self) -> int:
        return self.online_encoder.grid_w

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        z_context = self.online_encoder(context)
        pred = self.predictor(z_context)
        with torch.no_grad():
            z_target = self.target_encoder(target)
        if pred.shape != z_target.shape:
            raise RuntimeError(f"{self.name}: pred {tuple(pred.shape)} != target {tuple(z_target.shape)}")
        loss, terms, metrics = self.loss_fn(pred, z_target)
        return {
            "pred": pred,
            "target": z_target,
            "context": z_context,
            "loss": loss,
            "terms": terms,
            "metrics": metrics,
            "token_shape": tuple(z_context.shape),
        }

    @torch.no_grad()
    def update_target_ema(self, momentum: float) -> None:
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)
        for online_b, target_b in zip(self.online_encoder.buffers(), self.target_encoder.buffers()):
            target_b.copy_(online_b)


def _pool_tokens_2x2(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,D] tokens, got {tuple(tokens.shape)}")
    B, N, D = tokens.shape
    if N != grid_h * grid_w:
        raise ValueError(f"Token count {N} does not match grid {grid_h}x{grid_w}")
    if grid_h % 2 != 0 or grid_w % 2 != 0:
        raise ValueError(f"2x2 pooling requires even grid, got {grid_h}x{grid_w}")
    x = tokens.reshape(B, grid_h, grid_w, D)
    x = x.reshape(B, grid_h // 2, 2, grid_w // 2, 2, D).mean(dim=(2, 4))
    return x.reshape(B, (grid_h // 2) * (grid_w // 2), D)


def _cosine_align_tokens(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Cross-level token shapes must match, got {tuple(a.shape)} vs {tuple(b.shape)}")
    a_n = F.normalize(a, dim=-1, eps=1e-12)
    b_n = F.normalize(b.detach(), dim=-1, eps=1e-12)
    return 2.0 - 2.0 * (a_n * b_n).sum(dim=-1).mean()


def _default_hjepa_config() -> Dict[str, Any]:
    return {
        "levels": {
            "l1": {
                "patch_size": 8,
                "embed_dim": 192,
                "depth": 6,
                "num_heads": 6,
                "loss_type": "vicreg",
                "lambda_inv": 5.0,
                "lambda_var": 20.0,
                "lambda_cov": 2.0,
                "lambda_sig": 0.0,
                "weight": 1.0,
            },
            "l2": {
                "patch_size": 16,
                "embed_dim": 384,
                "depth": 10,
                "num_heads": 8,
                "loss_type": "vicreg_sigreg",
                "lambda_inv": 5.0,
                "lambda_var": 15.0,
                "lambda_cov": 1.0,
                "lambda_sig": 10.0,
                "weight": 1.0,
            },
            "l3": {
                "patch_size": 32,
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 8,
                "loss_type": "sigreg",
                "lambda_inv": 5.0,
                "lambda_var": 0.0,
                "lambda_cov": 0.0,
                "lambda_sig": 25.0,
                "weight": 1.0,
            },
        },
        "use_cross_level_loss": False,
        "lambda_cross_12": 0.5,
        "lambda_cross_23": 0.5,
        "cross_level_start_epoch": 5,
        "cross_level_warmup_epochs": 5,
        "ema_momentum": 0.996,
        "predictor_depth": 2,
        "predictor_hidden_mult": 4,
        "sigreg_eps": 1.0e-8,
        "frame_batch_size": 8,
    }


def _merge_hjepa_config(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(_default_hjepa_config())
    if user_cfg is None:
        return cfg
    for key, value in user_cfg.items():
        if key == "levels":
            for level_name, level_payload in value.items():
                if level_name not in cfg["levels"]:
                    raise ValueError(f"Unknown h_jepa level: {level_name}")
                cfg["levels"][level_name].update(level_payload)
        else:
            cfg[key] = value
    return cfg


class MultiScaleHJEPA(nn.Module):
    model_name = "h_jepa_multiscale"

    def __init__(self, cfg: "ViTJEPAConfig") -> None:
        super().__init__()
        hcfg = _merge_hjepa_config(cfg.h_jepa)
        self.h_jepa_config = hcfg
        self.ema_momentum = float(hcfg["ema_momentum"])
        self.use_cross_level_loss = bool(hcfg["use_cross_level_loss"])
        self.lambda_cross_12 = float(hcfg["lambda_cross_12"])
        self.lambda_cross_23 = float(hcfg["lambda_cross_23"])
        self.cross_level_start_epoch = int(hcfg["cross_level_start_epoch"])
        self.cross_level_warmup_epochs = int(hcfg["cross_level_warmup_epochs"])

        self.levels = nn.ModuleDict({
            name: HJEPALevel(
                name=name,
                level_cfg=level_cfg,
                img_size=cfg.img_size,
                mlp_ratio=cfg.mlp_ratio,
                drop_path_rate=cfg.drop_path_rate,
                predictor_depth=int(hcfg["predictor_depth"]),
                predictor_hidden_mult=int(hcfg["predictor_hidden_mult"]),
                sigreg_eps=float(hcfg["sigreg_eps"]),
                vicreg_eps=cfg.vicreg_eps,
                frame_batch_size=hcfg.get("frame_batch_size"),
            )
            for name, level_cfg in hcfg["levels"].items()
        })
        self.cross_proj_12 = (
            nn.Linear(self.levels["l1"].embed_dim, self.levels["l2"].embed_dim)
            if self.use_cross_level_loss else None
        )

    def _cross_weight(self, epoch: Optional[int]) -> float:
        if not self.use_cross_level_loss:
            return 0.0
        current_epoch = 0 if epoch is None else int(epoch)
        if current_epoch < self.cross_level_start_epoch:
            return 0.0
        warmup = max(self.cross_level_warmup_epochs, 1)
        return float(min(1.0, (current_epoch - self.cross_level_start_epoch + 1) / warmup))

    def forward(self, context: torch.Tensor, target: torch.Tensor, epoch: Optional[int] = None) -> Dict[str, Any]:
        level_out: Dict[str, Dict[str, Any]] = {}
        total = context.sum() * 0.0
        level_losses: Dict[str, torch.Tensor] = {}
        loss_terms: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, float] = {}
        token_shapes: Dict[str, Tuple[int, ...]] = {}

        for name, level in self.levels.items():
            out = level(context, target)
            level_out[name] = out
            total = total + level.weight * out["loss"]
            level_losses[f"{name}_loss"] = out["loss"]
            token_shapes[name] = out["token_shape"]
            for term_name, value in out["terms"].items():
                loss_terms[f"{name}_{term_name}"] = value
            for metric_name, value in out["metrics"].items():
                metrics[f"{name}_{metric_name}"] = value

        cross_weight = self._cross_weight(epoch)
        loss_cross_12 = total.detach() * 0.0
        loss_cross_23 = total.detach() * 0.0
        if cross_weight > 0.0:
            l1 = self.levels["l1"]
            l2 = self.levels["l2"]
            l1_pool = _pool_tokens_2x2(level_out["l1"]["context"], l1.grid_h, l1.grid_w)
            if self.cross_proj_12 is not None:
                l1_pool = self.cross_proj_12(l1_pool)
            loss_cross_12 = _cosine_align_tokens(l1_pool, level_out["l2"]["context"])
            l2_pool = _pool_tokens_2x2(level_out["l2"]["context"], l2.grid_h, l2.grid_w)
            loss_cross_23 = _cosine_align_tokens(l2_pool, level_out["l3"]["context"])
            total = total + cross_weight * (
                self.lambda_cross_12 * loss_cross_12
                + self.lambda_cross_23 * loss_cross_23
            )

        loss_terms["cross_12_loss"] = loss_cross_12
        loss_terms["cross_23_loss"] = loss_cross_23
        metrics["cross_weight"] = cross_weight
        metrics["cross_12_loss"] = float(loss_cross_12.detach().cpu())
        metrics["cross_23_loss"] = float(loss_cross_23.detach().cpu())
        metrics["loss_nonfinite"] = float(not torch.isfinite(total.detach()).item())

        return {
            "loss": total,
            "level_losses": level_losses,
            "loss_terms": loss_terms,
            "metrics": metrics,
            "token_shapes": token_shapes,
            "reprs": {
                name: {
                    "context": out["context"],
                    "target": out["target"],
                    "pred": out["pred"],
                }
                for name, out in level_out.items()
            },
        }

    @torch.no_grad()
    def update_target_ema(self, momentum: Optional[float] = None) -> None:
        m = self.ema_momentum if momentum is None else float(momentum)
        for level in self.levels.values():
            level.update_target_ema(m)

    @torch.no_grad()
    def extract_features(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        for name, level in self.levels.items():
            tokens = level.online_encoder(clip)
            feats[name] = tokens.mean(dim=1)
        feats["concat"] = torch.cat([feats["l1"], feats["l2"], feats["l3"]], dim=1)
        return feats

    def parameter_report(self) -> Dict[str, int]:
        report: Dict[str, int] = {}
        total = 0
        total_state = 0
        for name, level in self.levels.items():
            online = count_parameters(level.online_encoder)
            predictor = count_parameters(level.predictor)
            target = sum(p.numel() for p in level.target_encoder.parameters())
            level_trainable = online + predictor
            report[f"{name}_online_encoder_params"] = online
            report[f"{name}_target_encoder_params"] = target
            report[f"{name}_predictor_params"] = predictor
            report[f"{name}_params"] = level_trainable
            total += level_trainable
            total_state += level_trainable + target
        cross = count_parameters(self.cross_proj_12) if self.cross_proj_12 is not None else 0
        report["cross_level_params"] = cross
        report["total_params"] = total + cross
        report["total_state_params"] = total_state + cross
        return report


# ---------------------------------------------------------------------------
# Training entry point (formerly train.py)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ViTJEPAConfig:
    # ---- reproducibility ----
    seed: int = 42

    # ---- data ----
    data_root: str = "/scratch/${USER}/data/active_matter"
    output_dir: str = "/scratch/${USER}/am_jepa/vit_run"
    file_patterns: Tuple[str, ...] = ("*.hdf5", "*.h5")
    model_type: str = "physics_vit_jepa"

    # ---- windowing (must satisfy: (context_frames-1)*stride + gap + (target_frames-1)*stride < total_steps) ----
    context_frames: int = 16       # T for context clip — matches dataset native T
    target_frames: int = 16        # T for target clip
    frame_stride: int = 1          # stride between frames within a clip
    gap: int = 4                   # frames skipped between end of context and start of target
    window_start_stride: int = 4   # how often we sample windows along the trajectory
    max_windows_per_traj: Optional[int] = None

    # ---- training ----
    batch_size: int = 4            # per-GPU; effective = batch_size * world_size
    eval_batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 30
    warmup_epochs: int = 2         # linear warmup then constant LR

    lr: float = 1e-4               # fixed after warmup (no cosine decay)
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)

    # ---- VICReg loss (5/20/2 empirical recipe — do NOT change without ablation) ----
    lambda_inv: float = 5.0
    lambda_var: float = 20.0
    lambda_cov: float = 2.0
    vicreg_eps: float = 1e-4

    # ---- model architecture ----
    embed_dim: int = 384
    depth: int = 10
    num_heads: int = 8
    mlp_ratio: int = 4
    patch_size: int = 16
    img_size: int = 224
    drop_path_rate: float = 0.1
    predictor_width_mult: int = 4
    predictor_n_blocks: int = 2
    h_jepa: Dict[str, Any] = field(default_factory=dict)

    # ---- eval ----
    input_channels: int = 11       # kept for DataConfig compatibility
    spatial_eval_pool: int = 4     # adaptive avg pool before linear probe
    knn_k: int = 10

    # ---- collapse detection thresholds ----
    collapse_var_threshold: float = 1e-3
    collapse_rank_threshold: float = 5.0

    # ---- optimisation flags ----
    use_gradient_checkpointing: bool = True   # saves VRAM on A100
    compile_model: bool = False               # torch.compile; set True for A100

    # ---- bookkeeping ----
    stats_max_windows: int = 512
    save_every_steps: int = 500
    eval_every_epochs: int = 1
    log_every_steps: int = 25
    max_train_windows_per_split: Optional[int] = None
    max_eval_windows_per_split: Optional[int] = None
    smoke_max_windows_per_split: int = 16
    smoke_batches_per_epoch: int = 2

    # ---- wandb ----
    wandb_project: str = "active-matter-physics-vit-jepa"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

    # ---- resume ----
    resume: bool = True

    # ---- device (auto-set) ----
    device: str = "cuda"


def _to_data_config(cfg: ViTJEPAConfig) -> DataConfig:
    """Bridge our config to the DataConfig expected by dataloader utilities."""
    dc = DataConfig(
        seed=cfg.seed,
        data_root=cfg.data_root,
        output_dir=cfg.output_dir,
        model_type=("h_jepa_multiscale" if cfg.model_type == "h_jepa_multiscale" else "vit_spatial"),
        file_patterns=cfg.file_patterns,
        device=cfg.device,
        context_chunk_len=cfg.context_frames,
        target_chunk_len=cfg.target_frames,
        gap=cfg.gap,
        frame_stride=cfg.frame_stride,
        window_start_stride=cfg.window_start_stride,
        max_windows_per_traj=cfg.max_windows_per_traj,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        num_epochs=cfg.num_epochs,
        warmup_epochs=cfg.warmup_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        # VICReg — bridged; actual loss computed in this file
        lambda_inv=cfg.lambda_inv,
        lambda_var=cfg.lambda_var,
        lambda_cov=cfg.lambda_cov,
        vicreg_eps=cfg.vicreg_eps,
        embed_dim=cfg.embed_dim,
        input_channels=cfg.input_channels,
        latent_channels=cfg.embed_dim,
        spatial_eval_pool=cfg.spatial_eval_pool,
        knn_k=cfg.knn_k,
        collapse_var_threshold=cfg.collapse_var_threshold,
        collapse_rank_threshold=cfg.collapse_rank_threshold,
        stats_max_windows=cfg.stats_max_windows,
        save_every_steps=cfg.save_every_steps,
        eval_every_epochs=cfg.eval_every_epochs,
        log_every_steps=cfg.log_every_steps,
        max_train_windows_per_split=cfg.max_train_windows_per_split,
        max_eval_windows_per_split=cfg.max_eval_windows_per_split,
        smoke_max_windows_per_split=cfg.smoke_max_windows_per_split,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        wandb_mode=cfg.wandb_mode,
        resume=cfg.resume,
    )
    return dc


# ---------------------------------------------------------------------------
# VICReg loss — spatial, joint pred+target
# ---------------------------------------------------------------------------

def vicreg_loss(
    pred: torch.Tensor,
    z_target: torch.Tensor,
    cfg: ViTJEPAConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Flat VICReg over spatial tokens.

    pred, z_target: (B, D, H_p, W_p)

    The variance and covariance terms are computed on the JOINT distribution
    [pred_rows; z_target_rows] where each row is one spatial location from one
    sample.  This is the key design choice that prevents collapse — computing
    only on z_context is insufficient.

    Invariance term: cosine similarity loss (1 − cos) for unit-normalised
    representations.  This is more numerically stable than MSE for high-D
    vectors and naturally scale-invariant.
    """
    # Flatten spatial: (B, D, H, W) → (B*H*W, D)
    def _rows(z: torch.Tensor) -> torch.Tensor:
        B, D, H, W = z.shape
        return z.permute(0, 2, 3, 1).reshape(-1, D).float()

    pred_rows   = _rows(pred)
    target_rows = _rows(z_target)

    # --- Invariance (cosine) ---
    p_n = F.normalize(pred_rows,   dim=1, eps=1e-12)
    t_n = F.normalize(target_rows, dim=1, eps=1e-12)
    cos_sim  = (p_n * t_n).sum(dim=1)
    loss_inv = (1.0 - cos_sim).mean()

    # --- Joint distribution for variance + covariance ---
    joint = torch.cat([pred_rows, target_rows], dim=0)   # (2*B*H*W, D)

    # Variance: push each dimension to have std ≥ 1
    std = torch.sqrt(joint.var(dim=0, unbiased=False) + cfg.vicreg_eps)
    loss_var = F.relu(1.0 - std).mean()

    # Anchor mean near zero to prevent drift
    loss_mean = joint.mean(dim=0).pow(2).mean()
    loss_var  = loss_var + 0.1 * loss_mean

    # Covariance: off-diagonal penalty
    centered = joint - joint.mean(dim=0, keepdim=True)
    N, D     = centered.shape
    cov      = (centered.T @ centered) / max(N - 1, 1)      # (D, D)
    loss_cov = off_diagonal(cov).pow(2).sum() / D

    # --- Total ---
    loss_total = (
        cfg.lambda_inv * loss_inv
        + cfg.lambda_var * loss_var
        + cfg.lambda_cov * loss_cov
    )

    terms = {
        "loss_total": loss_total,
        "loss_inv":   loss_inv,
        "loss_var":   loss_var,
        "loss_cov":   loss_cov,
    }
    stats = collapse_stats(joint)
    stats["cosine_align_mean"] = float(cos_sim.mean().detach().cpu())
    return terms, stats


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed(device_str: str) -> Tuple[bool, int, int, int, str]:
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        return False, 0, 0, 1, device_str
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")
    local_rank = int(local_rank_env)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, local_rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parsing + config loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PhysicsViTJEPA on active matter.")
    p.add_argument("--config",          type=str, required=True)
    p.add_argument("--output_dir",      type=str, default=None)
    p.add_argument("--data_root",       type=str, default=None)
    p.add_argument("--wandb_project",   type=str, default=None)
    p.add_argument("--wandb_run_name",  type=str, default=None)
    p.add_argument("--wandb_mode",      choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--resume",          action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--compile",         action="store_true", help="Enable torch.compile")
    p.add_argument("--smoke_test",      action="store_true")
    p.add_argument("--eval_only",       action="store_true", help="Load checkpoint and run frozen probes only")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ViTJEPAConfig:
    with open(expand_path(args.config), "r") as f:
        payload = yaml.safe_load(f) or {}
    valid = set(ViTJEPAConfig.__dataclass_fields__.keys())
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    cfg = ViTJEPAConfig(**payload)
    if cfg.model_type not in {"physics_vit_jepa", "h_jepa_multiscale"}:
        raise ValueError(f"Unsupported model_type={cfg.model_type}")
    cfg.h_jepa = _merge_hjepa_config(cfg.h_jepa)
    for key in ["output_dir", "data_root", "wandb_project", "wandb_run_name",
                "wandb_mode", "resume"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    if args.compile:
        cfg.compile_model = True
    if args.eval_only:
        cfg.resume = True
    if args.smoke_test:
        cfg.num_epochs                   = 1
        cfg.max_train_windows_per_split  = cfg.smoke_max_windows_per_split
        cfg.max_eval_windows_per_split   = cfg.smoke_max_windows_per_split
        cfg.stats_max_windows            = min(cfg.stats_max_windows, 8)
        cfg.save_every_steps             = 2
        cfg.log_every_steps              = 1
        cfg.num_workers                  = 0
        if cfg.wandb_mode == "online":
            cfg.wandb_mode = "offline"
    if cfg.wandb_mode != "disabled" and wandb is None:
        raise ImportError("wandb required unless --wandb_mode disabled")
    return cfg


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: ViTJEPAConfig) -> nn.Module:
    if cfg.model_type == "h_jepa_multiscale":
        return MultiScaleHJEPA(cfg)
    model = PhysicsViTJEPA(
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        drop_path_rate=cfg.drop_path_rate,
        predictor_width_mult=cfg.predictor_width_mult,
        predictor_n_blocks=cfg.predictor_n_blocks,
    )
    if cfg.use_gradient_checkpointing:
        # Apply gradient checkpointing to transformer blocks
        for blk in model.encoder.blocks:
            blk.gradient_checkpointing = True
    return model


# ---------------------------------------------------------------------------
# Scheduler — linear warmup then constant LR
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ViTJEPAConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.warmup_epochs <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    constant = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, constant], milestones=[cfg.warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Collapse monitoring
# ---------------------------------------------------------------------------

def print_collapse_warning(stats: Dict[str, float], cfg: ViTJEPAConfig, prefix: str) -> bool:
    violated = (
        stats["var_min"] < cfg.collapse_var_threshold
        or stats["effective_rank"] < cfg.collapse_rank_threshold
    )
    if violated:
        print(
            f"!!! COLLAPSE WARNING [{prefix}] "
            f"var_min={stats['var_min']:.3e}  "
            f"eff_rank={stats['effective_rank']:.2f}  "
            f"align={stats.get('cosine_align_mean', float('nan')):.3f}"
        )
    return violated


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _config_signature(cfg: ViTJEPAConfig) -> Dict[str, Any]:
    ignored = {"output_dir", "data_root", "wandb_project", "wandb_run_name", "wandb_mode", "resume", "device"}
    return {k: v for k, v in asdict(cfg).items() if k not in ignored}


def _config_diffs(saved: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
    diffs: List[str] = []
    for key in sorted(set(saved) | set(current)):
        name = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            diffs.append(f"{name}: missing in checkpoint, current={current[key]!r}")
        elif key not in current:
            diffs.append(f"{name}: checkpoint={saved[key]!r}, missing now")
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            diffs.extend(_config_diffs(saved[key], current[key], name))
        elif saved[key] != current[key]:
            diffs.append(f"{name}: checkpoint={saved[key]!r}, current={current[key]!r}")
    return diffs


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: ViTJEPAConfig,
    epoch: int,
    global_step: int,
    best_valid_loss: float,
    history: List[Dict],
    run_id: Optional[str],
    collapse_violated: bool,
) -> None:
    payload = {
        "model_state":     unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch":           epoch,
        "global_step":     global_step,
        "best_valid_loss": best_valid_loss,
        "config":          asdict(cfg),
        "config_signature": _config_signature(cfg),
        "scaler_state":    None,
        "history":         history,
        "wandb_run_id":    run_id,
        "collapse_violated": bool(collapse_violated),
        "saved_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_torch_save(payload, path)
    print(f"✓ Checkpoint saved: {path}")
    if wandb is not None and wandb.run is not None:
        wandb.save(str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: ViTJEPAConfig,
    rank: int = 0,
) -> Tuple[int, int, float, List[Dict], Optional[str], bool]:
    if not cfg.resume or not path.exists():
        return 0, 0, float("inf"), [], None, False
    ckpt = torch.load(path, map_location="cpu")
    if "config_signature" in ckpt:
        diffs = _config_diffs(ckpt["config_signature"], _config_signature(cfg))
        if diffs:
            preview = "\n".join(f"  - {d}" for d in diffs[:20])
            more = "" if len(diffs) <= 20 else f"\n  ... {len(diffs) - 20} more"
            raise ValueError(
                f"Checkpoint config does not match current config:\n{preview}{more}\n"
                "Use --no-resume for an intentional architecture/config change."
            )
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch       = int(ckpt["epoch"]) + 1
    global_step       = int(ckpt["global_step"])
    best_valid_loss   = float(ckpt.get("best_valid_loss", float("inf")))
    history           = list(ckpt.get("history", []))
    run_id            = ckpt.get("wandb_run_id")
    collapse_violated = bool(ckpt.get("collapse_violated", False))
    if is_rank0(rank):
        print(f"Resumed from {path}: epoch={start_epoch} step={global_step} "
              f"best_loss={best_valid_loss:.6f}")
    return start_epoch, global_step, best_valid_loss, history, run_id, collapse_violated


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def _float_tensor(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _flatten_hjepa_output(output: Dict[str, Any], prefix: str) -> Dict[str, float]:
    payload: Dict[str, float] = {f"{prefix}/loss": _float_tensor(output["loss"])}
    for key, value in output["level_losses"].items():
        payload[f"{prefix}/{key}"] = _float_tensor(value)
    for key, value in output["loss_terms"].items():
        payload[f"{prefix}/{key}"] = _float_tensor(value)
    for key, value in output["metrics"].items():
        payload[f"{prefix}/{key}"] = float(value)
    return payload


def _hjepa_collapse_violated(metrics: Dict[str, float], cfg: ViTJEPAConfig, prefix: str, verbose: bool) -> bool:
    violated = False
    for level in ("l1", "l2", "l3"):
        var_min = float(metrics.get(f"{level}_var_min", float("nan")))
        eff_rank = float(metrics.get(f"{level}_effective_rank", float("nan")))
        norm_mean = float(metrics.get(f"{level}_repr_norm_mean", float("nan")))
        norm_std = float(metrics.get(f"{level}_repr_norm_std", float("nan")))
        nonfinite = bool(metrics.get(f"{level}_nonfinite", 0.0) or metrics.get(f"{level}_loss_nonfinite", 0.0))
        level_bad = (
            nonfinite
            or not np.isfinite(var_min)
            or not np.isfinite(eff_rank)
            or var_min < min(cfg.collapse_var_threshold, 1e-4)
            or eff_rank < cfg.collapse_rank_threshold
            or not np.isfinite(norm_mean)
            or norm_mean < 1e-4
            or norm_std < 1e-8
        )
        if level_bad:
            violated = True
            if verbose:
                print(
                    f"!!! H-JEPA COLLAPSE WARNING [{prefix}/{level}] "
                    f"var_min={var_min:.3e} eff_rank={eff_rank:.2f} "
                    f"norm_mean={norm_mean:.3e} norm_std={norm_std:.3e} nonfinite={int(nonfinite)}"
                )
    return violated


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    cfg: ViTJEPAConfig,
) -> Dict[str, float]:
    model.eval()
    if cfg.model_type == "h_jepa_multiscale":
        acc: Dict[str, List[float]] = {}
        for batch in tqdm(loader, desc="Valid", leave=False):
            batch = move_batch(batch, cfg.device)
            output = model(batch["context"], batch["target"], epoch=None)
            flat = _flatten_hjepa_output(output, "valid")
            for key, value in flat.items():
                acc.setdefault(key.replace("valid/", ""), []).append(float(value))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    acc: Dict[str, List[float]] = {k: [] for k in [
        "loss", "loss_inv", "loss_var", "loss_cov",
        "var_mean", "var_min", "effective_rank",
        "offdiag_cov_mean_abs", "cosine_align_mean",
    ]}
    for batch in tqdm(loader, desc="Valid", leave=False):
        batch = move_batch(batch, cfg.device)
        pred, z_target, _ = model(batch["context"], batch["target"])
        terms, stats = vicreg_loss(pred, z_target, cfg)
        acc["loss"].append(float(terms["loss_total"]))
        acc["loss_inv"].append(float(terms["loss_inv"]))
        acc["loss_var"].append(float(terms["loss_var"]))
        acc["loss_cov"].append(float(terms["loss_cov"]))
        for k in ["var_mean", "var_min", "effective_rank",
                  "offdiag_cov_mean_abs", "cosine_align_mean"]:
            acc[k].append(float(stats.get(k, 0.0)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    cfg: ViTJEPAConfig,
    output_dir: Path,
    rank: int = 0,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_config = _to_data_config(cfg)
    data_root   = expand_path(cfg.data_root)

    traj_df = assign_trajectory_splits(
        scan_trajectories(data_root, data_config, show_progress=is_rank0(rank)),
        cfg.seed,
    )
    if is_rank0(rank):
        traj_df.to_csv(output_dir / "trajectory_split.csv", index=False)
        print(traj_df.groupby("split").size().to_string())

    window_df = build_window_index(traj_df, data_config)
    if is_rank0(rank):
        window_df.to_csv(output_dir / "window_index.csv", index=False)

    if distributed and not is_rank0(rank):
        dist.barrier()
    stats = compute_channel_stats(
        window_df, data_config,
        output_dir / "train_channel_stats.npz",
        show_progress=is_rank0(rank),
    )
    if distributed and is_rank0(rank):
        dist.barrier()

    train_ds = ActiveMatterWindowDataset(
        window_df[window_df["split"] == "train"], data_config, stats,
        max_windows=cfg.max_train_windows_per_split,
    )
    valid_ds = ActiveMatterWindowDataset(
        window_df[window_df["split"] == "valid"], data_config, stats,
        max_windows=cfg.max_eval_windows_per_split,
    )
    test_ds = ActiveMatterWindowDataset(
        window_df[window_df["split"] == "test"], data_config, stats,
        max_windows=cfg.max_eval_windows_per_split,
    )

    sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    return (
        make_loader(train_ds, data_config, cfg.batch_size,   shuffle=True,  sampler=sampler),
        make_loader(valid_ds, data_config, cfg.eval_batch_size, shuffle=False),
        make_loader(test_ds,  data_config, cfg.eval_batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _ema_momentum_for_epoch(cfg: ViTJEPAConfig, model: nn.Module, epoch: int, batch_idx: int, batches_per_epoch: int) -> float:
    base = float(getattr(unwrap_model(model), "ema_momentum", 0.996))
    progress = (epoch + batch_idx / max(batches_per_epoch, 1)) / max(cfg.num_epochs, 1)
    return float(min(1.0, base + (1.0 - base) * progress))


def train_hjepa(
    cfg: ViTJEPAConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    output_dir: Path,
    start_epoch: int,
    global_step: int,
    best_valid_loss: float,
    history: List[Dict],
    run_id: Optional[str],
    collapse_violated: bool,
    smoke_test: bool,
    rank: int = 0,
    distributed: bool = False,
) -> Tuple[nn.Module, List[Dict], bool]:
    global SHUTDOWN_REQUESTED
    ckpt_dir = output_dir / "checkpoints"
    if is_rank0(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save(tag: str, epoch_value: int) -> None:
        save_checkpoint(
            ckpt_dir / f"{tag}.ckpt", model, optimizer, scheduler,
            cfg, epoch_value, global_step, best_valid_loss, history, run_id, collapse_violated,
        )

    printed_shapes = global_step > 0
    for epoch in range(start_epoch, cfg.num_epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        ep_acc: Dict[str, List[float]] = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                    disable=not is_rank0(rank))

        for batch_idx, batch in enumerate(pbar):
            if smoke_test and batch_idx >= cfg.smoke_batches_per_epoch:
                break
            batch = move_batch(batch, cfg.device)
            optimizer.zero_grad(set_to_none=True)

            output = model(batch["context"], batch["target"], epoch=epoch)
            if not printed_shapes and is_rank0(rank):
                print("H-JEPA token shapes:")
                for level_name, shape in output["token_shapes"].items():
                    print(f"  {level_name}: {shape}")
                printed_shapes = True

            loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"H-JEPA loss became non-finite at epoch={epoch} step={global_step}")
            loss.backward()
            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), cfg.grad_clip_norm)
            optimizer.step()

            raw_model = unwrap_model(model)
            if hasattr(raw_model, "update_target_ema"):
                raw_model.update_target_ema(_ema_momentum_for_epoch(cfg, model, epoch, batch_idx, len(train_loader)))
            global_step += 1

            train_payload = _flatten_hjepa_output(output, "train")
            for key, value in train_payload.items():
                ep_acc.setdefault(key.replace("train/", ""), []).append(float(value))

            if global_step % cfg.log_every_steps == 0 and is_rank0(rank):
                collapse_violated = (
                    _hjepa_collapse_violated(output["metrics"], cfg, "train", verbose=True)
                    or collapse_violated
                )
                train_payload.update({
                    "train/collapse_violated": int(collapse_violated),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "epoch": epoch,
                })
                if wandb is not None and wandb.run is not None:
                    wandb.log(train_payload, step=global_step)
                pbar.set_postfix(
                    loss=f"{train_payload['train/loss']:.4f}",
                    l1=f"{train_payload.get('train/l1_loss', float('nan')):.3f}",
                    l2=f"{train_payload.get('train/l2_loss', float('nan')):.3f}",
                    l3=f"{train_payload.get('train/l3_loss', float('nan')):.3f}",
                )

            if global_step % cfg.save_every_steps == 0 and is_rank0(rank):
                _save("last", epoch)
                _save(f"step_{global_step:06d}", epoch)

            if SHUTDOWN_REQUESTED:
                if is_rank0(rank):
                    _save("last", epoch)
                    print("Graceful shutdown after checkpoint.")
                sys.exit(0)

        scheduler.step()

        if is_rank0(rank):
            valid_metrics = run_validation(unwrap_model(model), valid_loader, cfg)
            collapse_violated = (
                _hjepa_collapse_violated(valid_metrics, cfg, "valid", verbose=True)
                or collapse_violated
            )
            row: Dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "collapse_violated": int(collapse_violated),
                "lr": scheduler.get_last_lr()[0],
            }
            row.update({f"train_{k.replace('/', '_')}": float(np.mean(v)) for k, v in ep_acc.items()})
            row.update({f"valid_{k.replace('/', '_')}": float(v) for k, v in valid_metrics.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

            valid_log = {f"valid/{k}": v for k, v in valid_metrics.items()}
            valid_log.update({
                "epoch": epoch,
                "global_step": global_step,
                "valid/collapse_violated": int(collapse_violated),
            })
            if wandb is not None and wandb.run is not None:
                wandb.log(valid_log, step=global_step)

            valid_loss = float(valid_metrics.get("loss", float("inf")))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                _save("best", epoch)
            _save("last", epoch)
            print(
                f"Epoch {epoch:03d} | train_loss={row.get('train_loss', float('nan')):.4f} "
                f"| valid_loss={valid_loss:.4f} | "
                f"l1_rank={valid_metrics.get('l1_effective_rank', float('nan')):.2f} "
                f"l2_rank={valid_metrics.get('l2_effective_rank', float('nan')):.2f} "
                f"l3_rank={valid_metrics.get('l3_effective_rank', float('nan')):.2f}"
            )

        if distributed:
            dist.barrier()

    best_path = ckpt_dir / "best.ckpt"
    if is_rank0(rank) and best_path.exists():
        ckpt = torch.load(best_path, map_location=cfg.device)
        unwrap_model(model).load_state_dict(ckpt["model_state"])
        collapse_violated = bool(ckpt.get("collapse_violated", collapse_violated))
    return model, history, collapse_violated


def train(
    cfg: ViTJEPAConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    output_dir: Path,
    start_epoch: int,
    global_step: int,
    best_valid_loss: float,
    history: List[Dict],
    run_id: Optional[str],
    collapse_violated: bool,
    smoke_test: bool,
    rank: int = 0,
    distributed: bool = False,
) -> Tuple[nn.Module, List[Dict], bool]:
    if cfg.model_type == "h_jepa_multiscale":
        return train_hjepa(
            cfg, model, optimizer, scheduler, train_loader, valid_loader, output_dir,
            start_epoch, global_step, best_valid_loss, history, run_id,
            collapse_violated, smoke_test, rank=rank, distributed=distributed,
        )

    global SHUTDOWN_REQUESTED
    ckpt_dir = output_dir / "checkpoints"
    if is_rank0(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save(tag: str) -> None:
        save_checkpoint(
            ckpt_dir / f"{tag}.ckpt", model, optimizer, scheduler,
            cfg, epoch, global_step, best_valid_loss, history, run_id, collapse_violated,
        )

    for epoch in range(start_epoch, cfg.num_epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        ep_losses: Dict[str, List[float]] = {k: [] for k in
            ["loss", "loss_inv", "loss_var", "loss_cov", "cosine_align_mean"]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                    disable=not is_rank0(rank))

        for batch_idx, batch in enumerate(pbar):
            if smoke_test and batch_idx >= cfg.smoke_batches_per_epoch:
                break

            batch = move_batch(batch, cfg.device)
            optimizer.zero_grad(set_to_none=True)

            pred, z_target, z_context = model(batch["context"], batch["target"])
            terms, stats = vicreg_loss(pred, z_target, cfg)

            # Log initial stats once to catch early collapse
            if global_step == 0 and is_rank0(rank):
                print(f"  [step 0] var_mean={stats['var_mean']:.4f}  "
                      f"var_min={stats['var_min']:.4f}  "
                      f"eff_rank={stats['effective_rank']:.2f}  "
                      f"align={stats['cosine_align_mean']:.4f}")

            terms["loss_total"].backward()
            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), cfg.grad_clip_norm)
            optimizer.step()
            global_step += 1

            for k in ep_losses:
                term_key = "loss_total" if k == "loss" else k
                ep_losses[k].append(
                    float(terms[term_key]) if term_key in terms else float(stats.get(k, 0.0))
                )

            if global_step % cfg.log_every_steps == 0 and is_rank0(rank):
                collapse_violated = print_collapse_warning(stats, cfg, "train") or collapse_violated
                log_payload = {
                    "train/loss":              float(terms["loss_total"]),
                    "train/loss_inv":          float(terms["loss_inv"]),
                    "train/loss_var":          float(terms["loss_var"]),
                    "train/loss_cov":          float(terms["loss_cov"]),
                    "train/var_mean":          stats["var_mean"],
                    "train/var_min":           stats["var_min"],
                    "train/eff_rank":          stats["effective_rank"],
                    "train/cosine_align":      stats["cosine_align_mean"],
                    "train/offdiag_cov":       stats["offdiag_cov_mean_abs"],
                    "train/collapse_violated": int(collapse_violated),
                    "train/lr":                optimizer.param_groups[0]["lr"],
                    "global_step":             global_step,
                    "epoch":                   epoch,
                }
                if wandb is not None and wandb.run is not None:
                    wandb.log(log_payload, step=global_step)
                pbar.set_postfix(
                    loss=f"{log_payload['train/loss']:.4f}",
                    rank=f"{log_payload['train/eff_rank']:.1f}",
                    align=f"{log_payload['train/cosine_align']:.3f}",
                )

            if global_step % cfg.save_every_steps == 0 and is_rank0(rank):
                _save("last")
                _save(f"step_{global_step:06d}")

            if SHUTDOWN_REQUESTED:
                if is_rank0(rank):
                    _save("last")
                    print("Graceful shutdown after checkpoint.")
                sys.exit(0)

        # ----- End of epoch -----
        scheduler.step()

        valid_metrics: Dict[str, float] = {}
        if is_rank0(rank):
            valid_metrics = run_validation(unwrap_model(model), valid_loader, cfg)

            # Check validation collapse
            if (valid_metrics["var_min"] < cfg.collapse_var_threshold or
                    valid_metrics["effective_rank"] < cfg.collapse_rank_threshold):
                collapse_violated = True
                print_collapse_warning(valid_metrics, cfg, "valid")

            row = {
                "epoch":            epoch,
                "global_step":      global_step,
                "train_loss":       float(np.mean(ep_losses["loss"])),
                "train_loss_inv":   float(np.mean(ep_losses["loss_inv"])),
                "train_loss_var":   float(np.mean(ep_losses["loss_var"])),
                "train_loss_cov":   float(np.mean(ep_losses["loss_cov"])),
                "train_cos_align":  float(np.mean(ep_losses["cosine_align_mean"])),
                "valid_loss":       valid_metrics.get("loss", float("nan")),
                "valid_loss_inv":   valid_metrics.get("loss_inv", float("nan")),
                "valid_loss_var":   valid_metrics.get("loss_var", float("nan")),
                "valid_loss_cov":   valid_metrics.get("loss_cov", float("nan")),
                "valid_eff_rank":   valid_metrics.get("effective_rank", float("nan")),
                "valid_var_min":    valid_metrics.get("var_min", float("nan")),
                "valid_cos_align":  valid_metrics.get("cosine_align_mean", float("nan")),
                "collapse_violated": int(collapse_violated),
                "lr":               scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(f"Epoch {epoch:03d} | " +
                  " | ".join(f"{k}={v:.4f}" for k, v in row.items()
                             if isinstance(v, float)))

            import pandas as pd
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

            valid_log = {f"valid/{k}": v for k, v in valid_metrics.items()}
            valid_log.update({"epoch": epoch, "global_step": global_step,
                               "valid/collapse_violated": int(collapse_violated)})
            if wandb is not None and wandb.run is not None:
                wandb.log(valid_log, step=global_step)

            if valid_metrics.get("loss", float("inf")) < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                _save("best")
            _save("last")

        if distributed:
            dist.barrier()

    # Restore best weights
    best_path = ckpt_dir / "best.ckpt"
    if is_rank0(rank) and best_path.exists():
        ckpt = torch.load(best_path, map_location=cfg.device)
        unwrap_model(model).load_state_dict(ckpt["model_state"])
        collapse_violated = bool(ckpt.get("collapse_violated", collapse_violated))

    return model, history, collapse_violated


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------

def install_signal_handlers(rank: int = 0) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        global SHUTDOWN_REQUESTED
        SHUTDOWN_REQUESTED = True
        if is_rank0(rank):
            print(f"Signal {signum} received — will checkpoint before exit.")
    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    cfg    = build_config(args)
    distributed, rank, local_rank, world_size, device = init_distributed(cfg.device)
    cfg.device = device

    cfg.output_dir = str(expand_path(cfg.output_dir))
    cfg.data_root  = str(expand_path(cfg.data_root))
    set_seed(cfg.seed + rank)
    install_signal_handlers(rank)

    output_dir = expand_path(cfg.output_dir)
    if is_rank0(rank):
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)
        atomic_json_save(asdict(cfg), output_dir / "config_resolved.json")
        print(json.dumps(asdict(cfg), indent=2))
        if distributed:
            print(f"DDP: world_size={world_size}  local_rank={local_rank}")
    if distributed:
        dist.barrier()

    # ---- Data ----
    train_loader, valid_loader, test_loader = prepare_data(
        cfg, output_dir, rank=rank, distributed=distributed
    )

    # ---- Model ----
    model = build_model(cfg).to(device)
    report = parameter_report(model)
    assert report["total_params"] < 100_000_000, \
        f"Model exceeds 100M parameter budget: {report['total_params']:,}"

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        if is_rank0(rank):
            print("torch.compile() enabled.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch, global_step, best_valid_loss, history, run_id, collapse_violated = \
        load_checkpoint(
            output_dir / "checkpoints" / "last.ckpt",
            model, optimizer, scheduler, cfg, rank,
        )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=(cfg.model_type == "h_jepa_multiscale"),
        )

    if is_rank0(rank):
        print(f"\n{cfg.model_type} parameter report:")
        for k, v in report.items():
            print(f"  {k:25s}: {v:>12,}")
        print()

    try:
        if is_rank0(rank) and cfg.wandb_mode != "disabled":
            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or cfg.model_type,
                id=run_id,
                resume="allow",
                mode=cfg.wandb_mode,
                config=asdict(cfg),
            )
            run_id = run.id
            wandb.config.update(report, allow_val_change=True)

        if args.eval_only:
            if global_step == 0 and start_epoch == 0:
                raise FileNotFoundError(
                    f"--eval_only requires a checkpoint at {output_dir / 'checkpoints' / 'last.ckpt'}"
                )
            if is_rank0(rank):
                print(f"Eval-only mode: using checkpoint at step {global_step}.")
        else:
            model, history, collapse_violated = train(
                cfg, model, optimizer, scheduler,
                train_loader, valid_loader, output_dir,
                start_epoch, global_step, best_valid_loss,
                history, run_id, collapse_violated,
                args.smoke_test, rank=rank, distributed=distributed,
            )

        if is_rank0(rank):
            data_cfg = _to_data_config(cfg)
            results_df, diagnostics_df = run_frozen_eval(
                data_cfg, model,
                train_loader, valid_loader, test_loader,
                output_dir, collapse_violated,
                smoke_test=args.smoke_test,
            )
            summary = {
                "model":            cfg.model_type,
                "output_dir":       str(output_dir.resolve()),
                "total_params":     report["total_params"],
                "encoder_params":   report.get("encoder_params"),
                "predictor_params": report.get("predictor_params"),
                "parameter_report": report,
                "collapse_violated": bool(collapse_violated),
                "results":          results_df.to_dict(orient="records"),
                "diagnostics":      diagnostics_df.to_dict(orient="records"),
            }
            atomic_json_save(summary, output_dir / "final_summary.json")
            if wandb is not None and wandb.run is not None:
                wandb.save(str(output_dir / "final_summary.json"))
                wandb.save(str(output_dir / "*.csv"))
            print(json.dumps(summary, indent=2))

    finally:
        if is_rank0(rank) and wandb is not None and wandb.run is not None:
            wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
