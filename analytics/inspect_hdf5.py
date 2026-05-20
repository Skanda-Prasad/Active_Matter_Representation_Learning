"""
active_matter_eda.py  (v2 — fixed loader)
==========================================
Complete EDA for polymathic-ai/active_matter.

Confirmed HDF5 structure (from inspect_hdf5.py):
  root attrs:  L, alpha, zeta, n_trajectories, ...
  t0_fields/concentration   (B, 81, 256, 256)
  t1_fields/velocity        (B, 81, 256, 256, 2)
  t2_fields/D               (B, 81, 256, 256, 2, 2)   <- orientation tensor
  t2_fields/E               (B, 81, 256, 256, 2, 2)   <- strain-rate tensor

11 channels after flattening:
  0          concentration
  1, 2       velocity  (x, y)
  3, 4, 5, 6 D_xx, D_xy, D_yx, D_yy
  7, 8, 9,10 E_xx, E_xy, E_yx, E_yy

Usage:
  python active_matter_eda.py --data_root /home/skanda/active_matter/data --split train --max_files 45
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── channel metadata ──────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    "conc",
    "vel_x", "vel_y",
    "D_xx", "D_xy", "D_yx", "D_yy",
    "E_xx", "E_xy", "E_yx", "E_yy",
]
CHANNEL_GROUPS = {
    "concentration": [0],
    "velocity":      [1, 2],
    "orientation D": [3, 4, 5, 6],
    "strain-rate E": [7, 8, 9, 10],
}
GROUP_COLORS = {
    "concentration": "#1D9E75",
    "velocity":      "#378ADD",
    "orientation D": "#EF9F27",
    "strain-rate E": "#D85A30",
}
N_CH = 11

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)


def ch_color(i):
    for grp, idxs in CHANNEL_GROUPS.items():
        if i in idxs:
            return GROUP_COLORS[grp]
    return "#888"


# ══════════════════════════════════════════════════════════════════════════════
# Loader  — returns (B, T, H, W, 11) float32
# ══════════════════════════════════════════════════════════════════════════════

def load_file(path):
    with h5py.File(path, "r") as f:
        alpha = float(f.attrs.get("alpha", f["scalars/alpha"][()]))
        zeta  = float(f.attrs.get("zeta",  f["scalars/zeta"][()]))
        L     = float(f.attrs.get("L",     f["scalars/L"][()]))

        conc = f["t0_fields/concentration"][()]   # (B, T, H, W)
        vel  = f["t1_fields/velocity"][()]        # (B, T, H, W, 2)
        D    = f["t2_fields/D"][()]               # (B, T, H, W, 2, 2)
        E    = f["t2_fields/E"][()]               # (B, T, H, W, 2, 2)

    B, T, H, W = conc.shape
    D_flat = D.reshape(B, T, H, W, 4)
    E_flat = E.reshape(B, T, H, W, 4)

    data = np.concatenate([
        conc[..., np.newaxis],
        vel,
        D_flat,
        E_flat,
    ], axis=-1).astype(np.float32)   # (B, T, H, W, 11)

    return data, {"alpha": alpha, "zeta": zeta, "L": L}


def collect_files(data_root, split):
    split_dir = Path(data_root) / split
    if split_dir.exists():
        files = sorted(split_dir.rglob("*.hdf5")) + sorted(split_dir.rglob("*.h5"))
    else:
        files = sorted(Path(data_root).rglob("*.hdf5")) + sorted(Path(data_root).rglob("*.h5"))
    return files


# ══════════════════════════════════════════════════════════════════════════════
# 1. Structure audit
# ══════════════════════════════════════════════════════════════════════════════

def audit_structure(files, max_files):
    print("\n[1/7] Structure audit ...")
    lines, param_list = [], []
    for fp in files[:max_files]:
        try:
            data, params = load_file(fp)
            lines.append(f"{fp.name}  shape={data.shape}  params={params}")
            param_list.append(params)
        except Exception as e:
            lines.append(f"{fp.name}  ERROR: {e}")
    (OUT_DIR / "structure_audit.txt").write_text("\n".join(lines))
    print(f"  {len(param_list)}/{len(files[:max_files])} files loaded OK  -> structure_audit.txt")
    return param_list


# ══════════════════════════════════════════════════════════════════════════════
# 2. Channel statistics & correlations
# ══════════════════════════════════════════════════════════════════════════════

def channel_statistics(files, max_files):
    print("\n[2/7] Channel statistics & correlations ...")
    all_means, all_stds, all_mins, all_maxs = [], [], [], []
    pixel_samples = []
    rng = np.random.default_rng(42)

    for fp in files[:max_files]:
        try:
            data, _ = load_file(fp)
            all_means.append(data.mean(axis=(0,1,2,3)))
            all_stds.append( data.std( axis=(0,1,2,3)))
            all_mins.append( data.min( axis=(0,1,2,3)))
            all_maxs.append( data.max( axis=(0,1,2,3)))
            flat = data.reshape(-1, N_CH)
            idx  = rng.integers(0, flat.shape[0], 3000)
            pixel_samples.append(flat[idx])
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    means = np.stack(all_means).mean(0)
    stds  = np.stack(all_stds ).mean(0)
    mins  = np.stack(all_mins ).min(0)
    maxs  = np.stack(all_maxs ).max(0)
    flat  = np.concatenate(pixel_samples, 0)
    corr  = np.corrcoef(flat.T)

    colors = [ch_color(i) for i in range(N_CH)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Channel statistics (averaged over all loaded files)", fontsize=13, fontweight="bold")
    for ax, vals, title in zip(axes.flat,
                                [means, stds, mins, maxs],
                                ["Mean", "Std dev", "Min", "Max"]):
        ax.bar(range(N_CH), vals, color=colors)
        ax.set_xticks(range(N_CH))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title)
    for grp, col in GROUP_COLORS.items():
        axes[0,0].bar(0, 0, color=col, label=grp)
    axes[0,0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "channel_stats.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    for i in range(N_CH):
        for j in range(N_CH):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr[i,j]) > 0.6 else "black")
    ax.set_title("Inter-channel pixel-level Pearson correlation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "channel_correlation.png", dpi=150)
    plt.close(fig)

    print("  -> channel_stats.png, channel_correlation.png")
    return {
        "means": means.tolist(), "stds": stds.tolist(),
        "mins":  mins.tolist(),  "maxs": maxs.tolist(),
        "corr_matrix": corr.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Temporal dynamics
# ══════════════════════════════════════════════════════════════════════════════

def temporal_dynamics(files, max_files):
    print("\n[3/7] Temporal dynamics ...")
    all_deltas, all_tvars, all_autocorr = [], [], []
    MAX_LAG = 20

    for fp in files[:max_files]:
        try:
            data, _ = load_file(fp)
            d = data.mean(0)              # (T, H, W, 11)
            T = d.shape[0]

            delta = np.abs(np.diff(d, axis=0)).mean(axis=(1,2))  # (T-1, 11)
            all_deltas.append(delta)
            all_tvars.append(d.var(axis=0).mean(axis=(0,1)))      # (11,)

            flat = d.reshape(T, -1)
            corrs = []
            for lag in range(1, min(MAX_LAG+1, T)):
                r = pearsonr(flat[:T-lag].ravel(), flat[lag:].ravel())[0]
                corrs.append(r)
            all_autocorr.append(corrs)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    mean_delta    = np.stack(all_deltas).mean(0)
    mean_tvar     = np.stack(all_tvars).mean(0)
    min_lag_len   = min(len(a) for a in all_autocorr)
    mean_autocorr = np.array([a[:min_lag_len] for a in all_autocorr]).mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Temporal dynamics", fontsize=13, fontweight="bold")
    ax = axes[0]
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(mean_delta[:, i], label=name, color=ch_color(i), lw=1.5, alpha=0.85)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Mean |delta| per channel")
    ax.set_title("Frame-to-frame change magnitude per channel")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    ax.bar(range(N_CH), mean_tvar, color=[ch_color(i) for i in range(N_CH)])
    ax.set_xticks(range(N_CH))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_title("Temporal variance per channel")
    ax.set_ylabel("Var over time (spatial mean)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_dynamics.png", dpi=150)
    plt.close(fig)

    lags = np.arange(1, min_lag_len + 1)
    drop_half  = next((int(l) for l, r in zip(lags, mean_autocorr) if r < 0.5), None)
    drop_tenth = next((int(l) for l, r in zip(lags, mean_autocorr) if r < 0.1), None)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lags, mean_autocorr, marker="o", color="#378ADD", lw=2, ms=5)
    ax.axhline(0.5, color="orange", lw=1, ls="--", label="r=0.5")
    ax.axhline(0.1, color="red",    lw=1, ls="--", label="r=0.1")
    ax.axhline(0,   color="gray",   lw=0.8)
    if drop_half:
        ax.axvline(drop_half,  color="orange", lw=1, ls=":")
        ax.text(drop_half+0.2, 0.52, f"lag {drop_half}", color="orange", fontsize=9)
    if drop_tenth:
        ax.axvline(drop_tenth, color="red",    lw=1, ls=":")
        ax.text(drop_tenth+0.2, 0.12, f"lag {drop_tenth}", color="red",    fontsize=9)
    ax.set_xlabel("Lag (frames, dt=0.25s)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Temporal autocorrelation — how many frames of context matter?", fontweight="bold")
    ax.set_xticks(lags); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_autocorr.png", dpi=150)
    plt.close(fig)

    print(f"  autocorr < 0.5 at lag {drop_half}, < 0.1 at lag {drop_tenth}")
    print("  -> temporal_dynamics.png, temporal_autocorr.png")
    return {
        "temporal_variance_per_channel": dict(zip(CHANNEL_NAMES, mean_tvar.tolist())),
        "most_dynamic_channel":  CHANNEL_NAMES[int(mean_tvar.argmax())],
        "least_dynamic_channel": CHANNEL_NAMES[int(mean_tvar.argmin())],
        "autocorr_below_0.5_at_lag":  drop_half,
        "autocorr_below_0.1_at_lag":  drop_tenth,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Power spectra
# ══════════════════════════════════════════════════════════════════════════════

def power_spectra(files, max_files):
    print("\n[4/7] Power spectra (2D FFT, radial average) ...")
    all_psd = []

    for fp in files[:max_files]:
        try:
            data, _ = load_file(fp)
            mid   = data.shape[1] // 2
            frame = data[:, mid].mean(0)   # (H, W, 11)
            H, W, C = frame.shape
            psds = []
            for c in range(C):
                f2d   = np.fft.fft2(frame[:,:,c])
                psd2d = np.abs(np.fft.fftshift(f2d)) ** 2
                cy, cx = H//2, W//2
                Y, X   = np.ogrid[:H, :W]
                r      = np.hypot(X-cx, Y-cy).astype(int)
                max_r  = min(cx, cy)
                radial = np.array([psd2d[r==ri].mean() if (r==ri).any() else 0.
                                   for ri in range(max_r)])
                psds.append(radial)
            min_len = min(len(p) for p in psds)
            all_psd.append(np.stack([p[:min_len] for p in psds], axis=-1))
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    min_len  = min(a.shape[0] for a in all_psd)
    mean_psd = np.stack([a[:min_len] for a in all_psd]).mean(0)   # (k, 11)
    k        = np.arange(1, min_len + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Radial power spectra per field group", fontsize=13, fontweight="bold")
    styles = ["-", "--", ":", "-."]
    for ax, (grp, idxs) in zip(axes.flat, CHANNEL_GROUPS.items()):
        ref_val = mean_psd[3, idxs[0]]
        for ii, i in enumerate(idxs):
            ax.loglog(k, mean_psd[:,i]+1e-30, label=CHANNEL_NAMES[i],
                      color=GROUP_COLORS[grp], lw=1.8, ls=styles[ii%4], alpha=0.9)
        ref_k = k[k>5]
        ax.loglog(ref_k, ref_val*(ref_k/ref_k[0])**(-5/3),
                  color="gray", lw=1, ls="--", label="k^-5/3 ref")
        ax.set_title(grp); ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Power")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "power_spectra.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("2D PSD (log scale) — anisotropy check", fontsize=11, fontweight="bold")
    try:
        data, _ = load_file(files[0])
        mid   = data.shape[1] // 2
        frame = data[:, mid].mean(0)
        for ax, (grp, idxs) in zip(axes, CHANNEL_GROUPS.items()):
            c     = idxs[0]
            f2d   = np.fft.fft2(frame[:,:,c])
            psd2d = np.log1p(np.abs(np.fft.fftshift(f2d))**2)
            ax.imshow(psd2d, cmap="inferno", interpolation="nearest")
            ax.set_title(f"{CHANNEL_NAMES[c]} ({grp})", fontsize=9)
            ax.axis("off")
    except Exception as e:
        print(f"  anisotropy plot skipped: {e}")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "spatial_anisotropy.png", dpi=150)
    plt.close(fig)

    peak_k = np.argmax(mean_psd, axis=0).tolist()
    print("  -> power_spectra.png, spatial_anisotropy.png")
    return {"peak_wavenumber_per_channel": dict(zip(CHANNEL_NAMES, peak_k))}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Phase diagram
# ══════════════════════════════════════════════════════════════════════════════

def phase_diagram(files, max_files):
    print("\n[5/7] Phase diagram (alpha x zeta) ...")
    records = defaultdict(list)

    for fp in files[:max_files]:
        try:
            data, params = load_file(fp)
            alpha, zeta = params["alpha"], params["zeta"]

            D_xx = data[:,:,:,:,3]
            D_xy = data[:,:,:,:,4]
            S    = (2 * np.sqrt((D_xx - 0.5)**2 + D_xy**2)).mean()

            vx, vy = data[:,:,:,:,1], data[:,:,:,:,2]
            KE     = (vx**2 + vy**2).mean()
            conc_var = data[:,:,:,:,0].var()

            records[(alpha, zeta)].append({"S": float(S), "KE": float(KE), "conc_var": float(conc_var)})
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    if not records:
        print("  ERROR: no records.")
        return {}

    alpha_vals = sorted(set(k[0] for k in records))
    zeta_vals  = sorted(set(k[1] for k in records))
    na, nz     = len(alpha_vals), len(zeta_vals)

    def make_grid(key):
        g = np.full((na, nz), np.nan)
        for (a, z), rlist in records.items():
            if a in alpha_vals and z in zeta_vals:
                g[alpha_vals.index(a), zeta_vals.index(z)] = np.mean([r[key] for r in rlist])
        return g

    S_grid  = make_grid("S")
    KE_grid = make_grid("KE")
    cv_grid = make_grid("conc_var")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase diagram: alpha x zeta", fontsize=13, fontweight="bold")
    for ax, grid, title, cmap in zip(
        axes.flat,
        [S_grid, KE_grid, cv_grid, S_grid],
        ["Scalar order param S (0=isotropic, 1=nematic)",
         "Mean kinetic energy |U|^2",
         "Concentration variance",
         "S (repeated for annotation reference)"],
        ["viridis", "plasma", "cividis", "viridis"]
    ):
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(range(nz))
        ax.set_xticklabels([f"z={z:.0f}" for z in zeta_vals], rotation=45, fontsize=8)
        ax.set_yticks(range(na))
        ax.set_yticklabels([f"a={a:.0f}" for a in alpha_vals], fontsize=9)
        ax.set_xlabel("zeta (alignment)"); ax.set_ylabel("alpha (dipole)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)
        for ai in range(na):
            for zi in range(nz):
                v = grid[ai, zi]
                if not np.isnan(v):
                    mean_v = np.nanmean(grid)
                    ax.text(zi, ai, f"{v:.3f}", ha="center", va="center",
                            fontsize=6.5, color="white" if v > mean_v else "black")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_diagram.png", dpi=150)
    plt.close(fig)

    # S profiles
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Order parameter profiles", fontsize=12, fontweight="bold")
    ax = axes[0]
    for ai, alpha in enumerate(alpha_vals):
        ax.plot(zeta_vals, S_grid[ai,:], marker="o", label=f"alpha={alpha:.0f}", lw=1.8)
    ax.set_xlabel("zeta"); ax.set_ylabel("S"); ax.set_title("S vs zeta for each alpha")
    ax.axhline(0.5, color="gray", lw=1, ls="--"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for zi, zeta in enumerate(zeta_vals):
        ax.plot(alpha_vals, S_grid[:,zi], marker="s", label=f"zeta={zeta:.0f}", lw=1.8)
    ax.set_xlabel("alpha"); ax.set_ylabel("S"); ax.set_title("S vs alpha for each zeta")
    ax.axhline(0.5, color="gray", lw=1, ls="--"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_profiles.png", dpi=150)
    plt.close(fig)

    transition = [
        {"alpha": alpha_vals[ai], "zeta": zeta_vals[zi], "S": float(S_grid[ai,zi])}
        for ai in range(na) for zi in range(nz)
        if not np.isnan(S_grid[ai,zi]) and 0.2 < S_grid[ai,zi] < 0.8
    ]
    print(f"  {len(transition)} (alpha,zeta) pairs in transition region (0.2 < S < 0.8)")
    print("  -> phase_diagram.png, phase_profiles.png")
    return {
        "alpha_values": alpha_vals, "zeta_values": zeta_vals,
        "S_range": [float(np.nanmin(S_grid)), float(np.nanmax(S_grid))],
        "transition_cells": transition,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Patch variance map
# ══════════════════════════════════════════════════════════════════════════════

def patch_statistics(files, max_files, patch_size=16):
    print(f"\n[6/7] Patch variance map (patch {patch_size}x{patch_size}) ...")
    P = 256 // patch_size
    patch_var_sum = np.zeros((P, P, N_CH))
    count = 0

    for fp in files[:max_files]:
        try:
            data, _ = load_file(fp)   # (B,T,256,256,11)
            var_map = data.var(axis=(0,1))   # (256,256,11)
            for pi in range(P):
                for pj in range(P):
                    r0,r1 = pi*patch_size, (pi+1)*patch_size
                    c0,c1 = pj*patch_size, (pj+1)*patch_size
                    patch_var_sum[pi,pj] += var_map[r0:r1,c0:c1].mean(axis=(0,1))
            count += 1
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    if count == 0:
        return {}

    mean_pv = patch_var_sum / count   # (P, P, 11)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Per-patch variance per channel (patch {patch_size}x{patch_size})",
                 fontsize=12, fontweight="bold")
    gv_min, gv_max = mean_pv.min(), mean_pv.max()
    for idx, (ax, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        im = ax.imshow(mean_pv[:,:,idx], cmap="hot", interpolation="nearest",
                       vmin=gv_min, vmax=gv_max)
        ax.set_title(name, fontsize=10); ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "patch_variance_per_channel.png", dpi=150)
    plt.close(fig)

    agg = mean_pv.mean(axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(agg, cmap="hot", interpolation="nearest")
    axes[0].set_title("Aggregate patch variance (all channels)")
    plt.colorbar(im, ax=axes[0])
    flat_pv = agg.ravel()
    axes[1].hist(flat_pv, bins=30, color="#1D9E75", edgecolor="white", lw=0.5)
    p25, p75 = np.percentile(flat_pv, 25), np.percentile(flat_pv, 75)
    axes[1].axvline(p25, color="red",  lw=1.5, ls="--", label=f"25th={p25:.4f}")
    axes[1].axvline(p75, color="blue", lw=1.5, ls="--", label=f"75th={p75:.4f}")
    axes[1].set_xlabel("Patch variance"); axes[1].set_ylabel("Count")
    axes[1].set_title("Patch variance distribution"); axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "patch_variance_aggregate.png", dpi=150)
    plt.close(fig)

    print("  -> patch_variance_per_channel.png, patch_variance_aggregate.png")
    return {
        "patch_var_mean": float(agg.mean()),
        "patch_var_std":  float(agg.std()),
        "high_var_fraction": float((agg > agg.mean()).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Per-parameter dynamics
# ══════════════════════════════════════════════════════════════════════════════

def per_param_dynamics(files, max_files):
    print("\n[7/7] Per-parameter dynamics ...")
    records = defaultdict(lambda: defaultdict(list))

    for fp in files[:max_files]:
        try:
            data, params = load_file(fp)
            alpha, zeta = params["alpha"], params["zeta"]
            d = data.mean(0)   # (T,H,W,11)

            records[(alpha,zeta)]["mean_delta"].append(
                float(np.abs(np.diff(d, axis=0)).mean()))
            records[(alpha,zeta)]["spatial_std"].append(
                float(d.std(axis=(1,2)).mean()))
            vx, vy = d[:,:,:,1], d[:,:,:,2]
            records[(alpha,zeta)]["vel_rms"].append(
                float(np.sqrt(vx**2 + vy**2).mean()))
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    alpha_vals = sorted(set(k[0] for k in records))
    zeta_vals  = sorted(set(k[1] for k in records))
    na, nz     = len(alpha_vals), len(zeta_vals)

    def make_grid(key):
        g = np.full((na, nz), np.nan)
        for (a, z), metrics in records.items():
            if a in alpha_vals and z in zeta_vals:
                g[alpha_vals.index(a), zeta_vals.index(z)] = np.mean(metrics[key])
        return g

    delta_grid = make_grid("mean_delta")
    sstd_grid  = make_grid("spatial_std")
    vel_grid   = make_grid("vel_rms")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Per-parameter dynamics", fontsize=12, fontweight="bold")
    for ax, grid, title, cmap in zip(
        axes,
        [delta_grid, sstd_grid, vel_grid],
        ["Frame-to-frame |delta| (evolution speed)",
         "Spatial heterogeneity (spatial std)",
         "Velocity RMS |U|"],
        ["YlOrRd", "Blues", "Greens"]
    ):
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(range(nz))
        ax.set_xticklabels([f"z={z:.0f}" for z in zeta_vals], rotation=45, fontsize=8)
        ax.set_yticks(range(na))
        ax.set_yticklabels([f"a={a:.0f}" for a in alpha_vals], fontsize=9)
        ax.set_xlabel("zeta"); ax.set_ylabel("alpha"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)
        for ai in range(na):
            for zi in range(nz):
                v = grid[ai,zi]
                if not np.isnan(v):
                    ax.text(zi, ai, f"{v:.4f}", ha="center", va="center",
                            fontsize=6, color="white" if v > np.nanmean(grid) else "black")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_param_dynamics.png", dpi=150)
    plt.close(fig)

    print("  -> per_param_dynamics.png")
    return {
        "fastest_alpha": alpha_vals[int(np.unravel_index(np.nanargmax(delta_grid), delta_grid.shape)[0])],
        "fastest_zeta":  zeta_vals[ int(np.unravel_index(np.nanargmax(delta_grid), delta_grid.shape)[1])],
        "slowest_alpha": alpha_vals[int(np.unravel_index(np.nanargmin(delta_grid), delta_grid.shape)[0])],
        "slowest_zeta":  zeta_vals[ int(np.unravel_index(np.nanargmin(delta_grid), delta_grid.shape)[1])],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--split",      default="train")
    parser.add_argument("--max_files",  type=int, default=45)
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()

    files = collect_files(args.data_root, args.split)
    print(f"Found {len(files)} HDF5 files in '{args.split}' split.")
    if not files:
        print("ERROR: no HDF5 files found.")
        return

    N = args.max_files
    summary = {"n_files_found": len(files), "n_files_analysed": min(N, len(files))}
    summary["structure"]          = audit_structure(files, N)
    summary["channel_stats"]      = channel_statistics(files, N)
    summary["temporal_dynamics"]  = temporal_dynamics(files, N)
    summary["power_spectra"]      = power_spectra(files, N)
    summary["phase_diagram"]      = phase_diagram(files, N)
    summary["patch_stats"]        = patch_statistics(files, N, args.patch_size)
    summary["per_param_dynamics"] = per_param_dynamics(files, N)

    out = OUT_DIR / "eda_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {OUT_DIR.resolve()}")
    print(f"Summary JSON: {out}")
    print(f"{'='*60}")
    print("\nKey files to open:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
