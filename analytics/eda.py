"""
active_matter_eda.py
====================
Complete exploratory data analysis for polymathic-ai/active_matter.

Covers:
  1. Dataset structure audit
  2. Channel statistics & inter-channel correlations
  3. Temporal dynamics (frame-to-frame delta, autocorrelation)
  4. Spatial structure (2D FFT power spectra per channel)
  5. Phase transition boundary (scalar order parameter S vs α, ζ)
  6. Patch-level statistics (useful for masking strategy design)
  7. Summary JSON of all findings

Usage:
  python active_matter_eda.py --data_root /path/to/active_matter --split train --max_files 20

Outputs (all saved to ./eda_outputs/):
  structure_audit.txt
  channel_stats.png
  channel_correlation.png
  temporal_dynamics.png
  temporal_autocorr.png
  power_spectra.png
  spatial_anisotropy.png
  phase_diagram.png
  patch_stats.png
  eda_summary.json
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Channel metadata ──────────────────────────────────────────────────────────
# 11 channels: 1 concentration (scalar) + 2 velocity (vector) +
#              4 orientation tensor (t2, symmetric) + 4 strain-rate tensor (t2)
CHANNEL_NAMES = [
    "conc",                          # 0  — concentration (scalar)
    "vel_x", "vel_y",                # 1,2 — velocity (vector)
    "orient_xx", "orient_xy",        # 3,4 — orientation tensor
    "orient_yx", "orient_yy",        # 5,6
    "strain_xx", "strain_xy",        # 7,8 — strain-rate tensor
    "strain_yx", "strain_yy",        # 9,10
]
CHANNEL_GROUPS = {
    "concentration": [0],
    "velocity":      [1, 2],
    "orientation":   [3, 4, 5, 6],
    "strain_rate":   [7, 8, 9, 10],
}
GROUP_COLORS = {
    "concentration": "#5DCAA5",
    "velocity":      "#378ADD",
    "orientation":   "#EF9F27",
    "strain_rate":   "#D85A30",
}

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def channel_color(ch_idx):
    for grp, idxs in CHANNEL_GROUPS.items():
        if ch_idx in idxs:
            return GROUP_COLORS[grp]
    return "#888"


def collect_hdf5_files(data_root: Path, split: str):
    """Find all HDF5 files for a given split directory."""
    split_dir = data_root / split
    if not split_dir.exists():
        # fall back: look for files with split name anywhere
        files = sorted(data_root.rglob(f"*{split}*.hdf5")) + \
                sorted(data_root.rglob(f"*{split}*.h5"))
    else:
        files = sorted(split_dir.rglob("*.hdf5")) + \
                sorted(split_dir.rglob("*.h5"))
    if not files:
        files = sorted(data_root.rglob("*.hdf5")) + \
                sorted(data_root.rglob("*.h5"))
    return files


def load_file(path: Path):
    """
    Load all field data from a Well-format HDF5 file.
    Returns:
        data: np.ndarray  shape (n_traj, T, H, W, 11)
        params: dict  {'alpha': float, 'zeta': float, ...}
    """
    with h5py.File(path, "r") as f:
        # ── parameters ──
        params = {}
        for key in f.attrs.get("simulation_parameters", []):
            params[key] = float(f.attrs[key])

        # ── fields ──
        # t0 = scalar, t1 = vector, t2 = tensor
        channels = []

        # concentration (scalar) → shape (B, T, H, W)
        if "t0_fields" in f:
            for name in f["t0_fields"].attrs.get("field_names", []):
                d = f["t0_fields"][name][:]          # (B,T,H,W) or (T,H,W)
                if d.ndim == 3:
                    d = d[np.newaxis]
                channels.append(d)                   # append (B,T,H,W)

        # velocity (vector) → shape (B, T, H, W, 2) → split to 2×(B,T,H,W)
        if "t1_fields" in f:
            for name in f["t1_fields"].attrs.get("field_names", []):
                d = f["t1_fields"][name][:]          # (B,T,H,W,2)
                if d.ndim == 4:
                    d = d[np.newaxis]
                for i in range(d.shape[-1]):
                    channels.append(d[..., i])

        # orientation + strain-rate (tensor) → (B,T,H,W,4) → 4×(B,T,H,W)
        if "t2_fields" in f:
            for name in f["t2_fields"].attrs.get("field_names", []):
                d = f["t2_fields"][name][:]          # (B,T,H,W,D^2)
                if d.ndim == 4:
                    d = d[np.newaxis]
                for i in range(d.shape[-1]):
                    channels.append(d[..., i])

        if not channels:
            raise ValueError(f"No recognisable field groups in {path}")

        # Stack → (B, T, H, W, C)
        # All channels should have same (B,T,H,W)
        data = np.stack(channels, axis=-1).astype(np.float32)

    return data, params


# ══════════════════════════════════════════════════════════════════════════════
# 1. Structure audit
# ══════════════════════════════════════════════════════════════════════════════

def audit_structure(files, max_files, out_dir):
    print("\n[1/8] Structure audit ...")
    lines = []
    param_sets = []

    for fpath in files[:max_files]:
        try:
            data, params = load_file(fpath)
            B, T, H, W, C = data.shape
            lines.append(
                f"{fpath.name:50s}  shape={data.shape}  "
                f"dtype={data.dtype}  params={params}"
            )
            param_sets.append(params)
        except Exception as e:
            lines.append(f"{fpath.name:50s}  ERROR: {e}")

    report = "\n".join(lines)
    (out_dir / "structure_audit.txt").write_text(report)
    print(f"  Audited {len(lines)} files → structure_audit.txt")
    return param_sets


# ══════════════════════════════════════════════════════════════════════════════
# 2. Channel statistics & correlations
# ══════════════════════════════════════════════════════════════════════════════

def channel_statistics(files, max_files, out_dir):
    print("\n[2/8] Channel statistics & correlations ...")
    all_means, all_stds, all_mins, all_maxs = [], [], [], []
    flat_samples = []   # for correlation: (N_pixels_sampled, 11)

    rng = np.random.default_rng(42)

    for fpath in files[:max_files]:
        try:
            data, _ = load_file(fpath)          # (B,T,H,W,C)
            B, T, H, W, C = data.shape
            all_means.append(data.mean(axis=(0, 1, 2, 3)))   # (C,)
            all_stds.append(data.std(axis=(0, 1, 2, 3)))
            all_mins.append(data.min(axis=(0, 1, 2, 3)))
            all_maxs.append(data.max(axis=(0, 1, 2, 3)))
            # sample 2000 random pixels for correlation
            idx = rng.integers(0, B * T * H * W, size=2000)
            flat = data.reshape(-1, C)[idx]
            flat_samples.append(flat)
        except Exception:
            continue

    means = np.stack(all_means).mean(0)
    stds  = np.stack(all_stds).mean(0)
    mins  = np.stack(all_mins).min(0)
    maxs  = np.stack(all_maxs).max(0)
    flat  = np.concatenate(flat_samples, axis=0)   # (N, 11)

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Channel statistics across dataset", fontsize=14, fontweight="bold")

    colors = [channel_color(i) for i in range(len(CHANNEL_NAMES))]

    for ax, vals, title in zip(
        axes.flat,
        [means, stds, mins, maxs],
        ["Mean", "Std dev", "Min", "Max"]
    ):
        bars = ax.bar(range(len(CHANNEL_NAMES)), vals, color=colors)
        ax.set_xticks(range(len(CHANNEL_NAMES)))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)

    # legend
    for grp, col in GROUP_COLORS.items():
        axes[0, 0].bar(0, 0, color=col, label=grp)
    axes[0, 0].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "channel_stats.png", dpi=150)
    plt.close(fig)

    # ── correlation matrix ──
    corr = np.corrcoef(flat.T)   # (11, 11)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CHANNEL_NAMES)))
    ax.set_yticks(range(len(CHANNEL_NAMES)))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    for i in range(len(CHANNEL_NAMES)):
        for j in range(len(CHANNEL_NAMES)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr[i,j]) < 0.6 else "white")
    ax.set_title("Inter-channel correlation (pixel-level)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "channel_correlation.png", dpi=150)
    plt.close(fig)

    print("  → channel_stats.png, channel_correlation.png")
    return {"means": means.tolist(), "stds": stds.tolist(),
            "mins": mins.tolist(), "maxs": maxs.tolist(),
            "corr_matrix": corr.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Temporal dynamics
# ══════════════════════════════════════════════════════════════════════════════

def temporal_dynamics(files, max_files, out_dir):
    print("\n[3/8] Temporal dynamics ...")
    # frame-to-frame L2 delta per channel, averaged over space
    all_deltas = []   # list of (T-1, C) arrays
    all_vars   = []   # temporal variance per traj: (T, C)

    for fpath in files[:max_files]:
        try:
            data, _ = load_file(fpath)       # (B,T,H,W,C)
            # average over batch dim
            data = data.mean(0)              # (T,H,W,C)
            T, H, W, C = data.shape
            # frame delta: L2 per channel
            delta = np.abs(np.diff(data, axis=0)).mean(axis=(1, 2))  # (T-1, C)
            all_deltas.append(delta)
            # temporal variance: how much does each spatial pixel vary over time?
            tvar = data.var(axis=0).mean(axis=(0, 1))   # (C,)
            all_vars.append(tvar)
        except Exception:
            continue

    mean_delta = np.stack(all_deltas).mean(0)   # (T-1, C)
    mean_tvar  = np.stack(all_vars).mean(0)      # (C,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temporal dynamics", fontsize=13, fontweight="bold")

    # left: frame delta over time
    ax = axes[0]
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(mean_delta[:, i], label=name, color=channel_color(i),
                alpha=0.85, lw=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean |Δ| per channel")
    ax.set_title("Frame-to-frame change magnitude")
    ax.legend(fontsize=7, ncol=2)

    # right: temporal variance per channel
    ax = axes[1]
    colors = [channel_color(i) for i in range(len(CHANNEL_NAMES))]
    ax.bar(range(len(CHANNEL_NAMES)), mean_tvar, color=colors)
    ax.set_xticks(range(len(CHANNEL_NAMES)))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_title("Temporal variance per channel\n(how much each channel evolves)")
    ax.set_ylabel("Variance over time (spatial mean)")

    plt.tight_layout()
    fig.savefig(out_dir / "temporal_dynamics.png", dpi=150)
    plt.close(fig)

    # ── autocorrelation per channel ──
    # compute temporal autocorrelation: does frame t predict frame t+lag?
    all_autocorr = []
    max_lag = min(15, mean_delta.shape[0])

    for fpath in files[:max_files]:
        try:
            data, _ = load_file(fpath)
            data = data.mean(0)           # (T,H,W,C)
            T, H, W, C = data.shape
            flat = data.reshape(T, -1)    # (T, H*W*C)
            lags, corrs = [], []
            for lag in range(1, max_lag + 1):
                r = pearsonr(flat[:T-lag].ravel(), flat[lag:].ravel())[0]
                lags.append(lag)
                corrs.append(r)
            all_autocorr.append(corrs)
        except Exception:
            continue

    mean_autocorr = np.array(all_autocorr).mean(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, max_lag + 1), mean_autocorr, marker="o", color="#378ADD", lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Pearson r (frame t vs t+lag)")
    ax.set_title("Temporal autocorrelation — how many frames of context matter?",
                 fontweight="bold")
    ax.set_xticks(range(1, max_lag + 1))
    plt.tight_layout()
    fig.savefig(out_dir / "temporal_autocorr.png", dpi=150)
    plt.close(fig)

    print("  → temporal_dynamics.png, temporal_autocorr.png")
    return {
        "mean_temporal_variance_per_channel": mean_tvar.tolist(),
        "most_dynamic_channel": CHANNEL_NAMES[int(mean_tvar.argmax())],
        "least_dynamic_channel": CHANNEL_NAMES[int(mean_tvar.argmin())],
        "temporal_autocorr_at_lag1": float(mean_autocorr[0]),
        "temporal_autocorr_at_lag4": float(mean_autocorr[3]) if len(mean_autocorr) > 3 else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Spatial structure — 2D FFT power spectra
# ══════════════════════════════════════════════════════════════════════════════

def power_spectra(files, max_files, out_dir):
    print("\n[4/8] Power spectra (2D FFT) ...")
    # Average radial power spectrum per channel
    all_psd = []   # list of (n_wavenumbers, C) arrays

    for fpath in files[:max_files]:
        try:
            data, _ = load_file(fpath)       # (B,T,H,W,C)
            # use middle frame, average over batch
            mid = data.shape[1] // 2
            frame = data[:, mid, :, :, :].mean(0)   # (H,W,C)
            H, W, C = frame.shape
            psds = []
            for c in range(C):
                f2d = np.fft.fft2(frame[:, :, c])
                psd2d = np.abs(np.fft.fftshift(f2d)) ** 2
                # radial average
                cy, cx = H // 2, W // 2
                Y, X = np.ogrid[:H, :W]
                r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                max_r = min(cx, cy)
                radial = np.array([psd2d[r == ri].mean() if (r == ri).any() else 0
                                   for ri in range(max_r)])
                psds.append(radial)
            # trim to min length
            min_len = min(len(p) for p in psds)
            all_psd.append(np.stack([p[:min_len] for p in psds], axis=-1))
        except Exception:
            continue

    min_len = min(a.shape[0] for a in all_psd)
    mean_psd = np.stack([a[:min_len] for a in all_psd]).mean(0)  # (k, C)
    k = np.arange(1, min_len + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Radial power spectra per channel group", fontsize=13, fontweight="bold")

    for ax, (grp, idxs) in zip(axes.flat, CHANNEL_GROUPS.items()):
        for i in idxs:
            psd = mean_psd[:, i]
            ax.loglog(k, psd + 1e-30, label=CHANNEL_NAMES[i],
                      color=GROUP_COLORS[grp], alpha=0.85,
                      lw=1.5, ls=["-","--",":","-."][idxs.index(i) % 4])
        ax.set_title(f"{grp} channels")
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Power")
        ax.legend(fontsize=8)
        # reference slope k^-5/3 (Kolmogorov)
        ref_k = k[k > 3]
        ax.loglog(ref_k, ref_k.astype(float) ** (-5/3) * mean_psd[3, idxs[0]],
                  color="gray", lw=1, ls="--", label="k^-5/3")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "power_spectra.png", dpi=150)
    plt.close(fig)

    # ── spatial anisotropy: is the PSD isotropic or directional? ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("2D power spectrum (middle frame, file 0) — anisotropy check",
                 fontsize=12, fontweight="bold")
    try:
        data, _ = load_file(files[0])
        mid = data.shape[1] // 2
        frame = data[:, mid, :, :, :].mean(0)
        H, W, C = frame.shape
        for ax, (grp, idxs) in zip(axes, CHANNEL_GROUPS.items()):
            c = idxs[0]
            f2d = np.fft.fft2(frame[:, :, c])
            psd2d = np.log1p(np.abs(np.fft.fftshift(f2d)) ** 2)
            ax.imshow(psd2d, cmap="inferno", interpolation="nearest")
            ax.set_title(f"{CHANNEL_NAMES[c]}", fontsize=9)
            ax.axis("off")
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(out_dir / "spatial_anisotropy.png", dpi=150)
    plt.close(fig)

    print("  → power_spectra.png, spatial_anisotropy.png")
    return {"dominant_scale_per_channel": int(np.argmax(mean_psd, axis=0).mean())}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Phase transition boundary — scalar order parameter S vs (α, ζ)
# ══════════════════════════════════════════════════════════════════════════════

def phase_diagram(files, max_files, out_dir):
    """
    Scalar order parameter S = <|2*D_xx - 1|> where D is orientation tensor.
    S ≈ 0 → isotropic,  S ≈ 1 → fully nematic.
    We also track: mean |velocity|, concentration variance.
    """
    print("\n[5/8] Phase diagram (order parameter S vs α, ζ) ...")
    results = []   # list of {alpha, zeta, S, mean_speed, conc_var, ...}

    orient_xx_idx = 3   # D_xx component of orientation tensor

    for fpath in files[:max_files]:
        try:
            data, params = load_file(fpath)  # (B,T,H,W,C)
            alpha = params.get("alpha", params.get("Alpha", None))
            zeta  = params.get("zeta",  params.get("Zeta",  None))
            if alpha is None or zeta is None:
                # try to infer from filename or other attrs
                continue

            # scalar order parameter: S = mean over space/time of |2*D_xx - 1|
            d_xx = data[:, :, :, :, orient_xx_idx]  # (B,T,H,W)
            S = np.abs(2 * d_xx - 1).mean()

            # mean speed |U|
            vx = data[:, :, :, :, 1]
            vy = data[:, :, :, :, 2]
            speed = np.sqrt(vx**2 + vy**2).mean()

            # concentration variance
            conc = data[:, :, :, :, 0]
            conc_var = conc.var()

            # enstrophy proxy: var of velocity divergence
            # approximate: var(vx - vy) across space
            div_proxy = (vx - vy).var()

            results.append({
                "alpha": float(alpha), "zeta": float(zeta),
                "S": float(S), "speed": float(speed),
                "conc_var": float(conc_var), "div_proxy": float(div_proxy),
            })
        except Exception as e:
            continue

    if not results:
        print("  WARNING: Could not extract alpha/zeta params from any file.")
        print("  → Skipping phase diagram. Check your HDF5 attribute names.")
        return {}

    # group by (alpha, zeta) and average
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["alpha"], r["zeta"])].append(r)

    alpha_vals = sorted(set(r["alpha"] for r in results))
    zeta_vals  = sorted(set(r["zeta"]  for r in results))

    def make_grid(key):
        grid = np.full((len(alpha_vals), len(zeta_vals)), np.nan)
        for (a, z), rlist in grouped.items():
            if a in alpha_vals and z in zeta_vals:
                ai = alpha_vals.index(a)
                zi = zeta_vals.index(z)
                grid[ai, zi] = np.mean([r[key] for r in rlist])
        return grid

    S_grid     = make_grid("S")
    speed_grid = make_grid("speed")
    cvar_grid  = make_grid("conc_var")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase diagram — parameter space (α × ζ)", fontsize=13, fontweight="bold")

    for ax, grid, title, cmap in zip(
        axes,
        [S_grid, speed_grid, cvar_grid],
        ["Scalar order param S\n(0=isotropic, 1=nematic)",
         "Mean speed |U|",
         "Concentration variance"],
        ["viridis", "plasma", "cividis"]
    ):
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower",
                       interpolation="nearest")
        ax.set_xticks(range(len(zeta_vals)))
        ax.set_xticklabels([f"ζ={z:.0f}" for z in zeta_vals], rotation=45, fontsize=8)
        ax.set_yticks(range(len(alpha_vals)))
        ax.set_yticklabels([f"α={a:.0f}" for a in alpha_vals], fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(out_dir / "phase_diagram.png", dpi=150)
    plt.close(fig)

    print("  → phase_diagram.png")
    return {
        "alpha_values_found": alpha_vals,
        "zeta_values_found": zeta_vals,
        "S_range": [float(np.nanmin(S_grid)), float(np.nanmax(S_grid))],
        "phase_transition_notes": (
            "S close to 0 = isotropic phase. S close to 1 = nematic. "
            "Look for rapid change in S across the ζ axis — that boundary "
            "is the hardest region for the model and most informative for representations."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Patch-level statistics (masking strategy design)
# ══════════════════════════════════════════════════════════════════════════════

def patch_statistics(files, max_files, out_dir, patch_size=16):
    """
    Compute per-patch variance across the spatial grid.
    High-variance patches = more informative → good masking targets.
    Low-variance patches = background / near-uniform → less useful to reconstruct.
    """
    print("\n[6/8] Patch-level statistics (masking strategy) ...")
    H_pat = W_pat = 224 // patch_size
    all_patch_vars = np.zeros((H_pat, W_pat))
    count = 0

    for fpath in files[:max_files]:
        try:
            data, _ = load_file(fpath)       # (B,T,H,W,C)
            B, T, H, W, C = data.shape
            # resize to 224 if needed
            if H != 224 or W != 224:
                # simple centre crop to nearest multiple of patch_size
                h_start = (H - 224) // 2
                w_start = (W - 224) // 2
                if h_start >= 0 and w_start >= 0:
                    data = data[:, :, h_start:h_start+224, w_start:w_start+224, :]
                else:
                    continue

            # average over batch + time + channels → (224, 224) variance map
            var_map = data.var(axis=(0, 1, 4))   # (H, W)

            # pool into patches
            for pi in range(H_pat):
                for pj in range(W_pat):
                    r0, r1 = pi * patch_size, (pi + 1) * patch_size
                    c0, c1 = pj * patch_size, (pj + 1) * patch_size
                    all_patch_vars[pi, pj] += var_map[r0:r1, c0:c1].mean()
            count += 1
        except Exception:
            continue

    if count == 0:
        print("  WARNING: Could not compute patch stats (check spatial resolution).")
        return {}

    mean_patch_vars = all_patch_vars / count

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Patch variance map (patch size {patch_size}×{patch_size})",
                 fontsize=12, fontweight="bold")

    im = axes[0].imshow(mean_patch_vars, cmap="hot", interpolation="nearest")
    axes[0].set_title("Mean variance per patch\n(brighter = more informative)")
    plt.colorbar(im, ax=axes[0])

    flat = mean_patch_vars.ravel()
    axes[1].hist(flat, bins=30, color="#5DCAA5", edgecolor="white", lw=0.5)
    axes[1].axvline(np.percentile(flat, 25), color="red", lw=1.5, ls="--",
                    label="25th pct (easy mask candidates)")
    axes[1].axvline(np.percentile(flat, 75), color="blue", lw=1.5, ls="--",
                    label="75th pct (hard mask candidates)")
    axes[1].set_xlabel("Patch variance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Patch variance distribution")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "patch_stats.png", dpi=150)
    plt.close(fig)

    print("  → patch_stats.png")
    return {
        "patch_var_mean": float(mean_patch_vars.mean()),
        "patch_var_std": float(mean_patch_vars.std()),
        "high_var_fraction": float((mean_patch_vars > mean_patch_vars.mean()).mean()),
        "insight": (
            "Patches with above-average variance are prime masking targets — "
            "predicting them forces the model to learn real dynamics. "
            "Consider bias-masking toward high-variance patches."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Active matter EDA")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the active_matter dataset")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to analyse (train/valid/test)")
    parser.add_argument("--max_files", type=int, default=20,
                        help="Max HDF5 files to load (keep low for speed)")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size for patch statistics")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = collect_hdf5_files(data_root, args.split)
    print(f"Found {len(files)} HDF5 files in '{args.split}' split.")
    if not files:
        print("ERROR: No HDF5 files found. Check --data_root path.")
        return

    summary = {"n_files_found": len(files), "max_files_analysed": args.max_files}

    summary["params_found"]      = audit_structure(files, args.max_files, OUT_DIR)
    summary["channel_stats"]     = channel_statistics(files, args.max_files, OUT_DIR)
    summary["temporal_dynamics"] = temporal_dynamics(files, args.max_files, OUT_DIR)
    summary["power_spectra"]     = power_spectra(files, args.max_files, OUT_DIR)
    summary["phase_diagram"]     = phase_diagram(files, args.max_files, OUT_DIR)
    summary["patch_stats"]       = patch_statistics(files, args.max_files, OUT_DIR,
                                                    args.patch_size)

    # ── Write JSON summary ──
    summary_path = OUT_DIR / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"EDA complete. All outputs in: {OUT_DIR.resolve()}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    print("\nKey outputs to review:")
    print("  channel_stats.png      — which channels have signal, which are near-zero")
    print("  channel_correlation.png — which channels are redundant vs independent")
    print("  temporal_dynamics.png  — which channels change fastest (mask priority)")
    print("  temporal_autocorr.png  — how many frames of context actually matter")
    print("  power_spectra.png      — scale of dominant spatial structures")
    print("  spatial_anisotropy.png — whether spatial structure is isotropic")
    print("  phase_diagram.png      — where the isotropic→nematic transition sits")
    print("  patch_stats.png        — which spatial regions are most informative")
    print("  eda_summary.json       — machine-readable summary of all findings")


if __name__ == "__main__":
    main()
