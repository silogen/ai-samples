# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
8-panel precipitation comparison figure (2 rows × 4 columns).

Row 1 (1.0° inputs):
  Col 1: ERA5 initial precipitation  2020-01-01  (1.0°)
  Col 2: GenCast forecast            2020-01-02  (1.0°)
  Col 3: ERA5 ground truth           2020-01-02  (1.0°)
  Col 4: ORBIT-2 downscaled output                (0.25° — higher res)

Row 2 (0.25° inputs):
  Col 1: ERA5 initial precipitation  2020-01-01  (0.25°)
  Col 2: GenCast forecast            2020-01-02  (0.25°)
  Col 3: ERA5 ground truth           2020-01-02  (0.25°)
  Col 4: ORBIT-2 downscaled output                (higher res)

ERA5 cols 1 & 3 are downloaded from CDS (cached as .npy after first run).

Usage:
    conda run -n vis-wp python plot_gencast.py
    conda run -n vis-wp python plot_gencast.py --no-cds --out results/gencast_comparison.png
"""

import argparse
import os
import tempfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

try:
    import cmocean
    CMAP = cmocean.cm.rain
except ImportError:
    CMAP = plt.cm.YlGnBu

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out",    default="results/gencast_comparison.png")
parser.add_argument("--no-cds", action="store_true",
                    help="Skip CDS download; use cached ERA5 .npy files only")
parser.add_argument("--all-cols", action="store_true",
                    help="Show all 4 columns (ERA5 initial, GenCast, ERA5 truth, downscaled). "
                         "Default: show only GenCast forecast + downscaled (cols 2 & 4)")
parser.add_argument("--results", default="results")
args = parser.parse_args()

R = args.results   # shorthand

# ── CDS download helper ───────────────────────────────────────────────────────
def download_era5_tp(date_str: str, cache_path: str, no_cds: bool) -> np.ndarray:
    """Return ERA5 24h total precipitation (m) at 0.25° on date_str (YYYY-MM-DD).

    Downloads from CDS on first call; subsequent calls load the cached .npy.
    Latitude order: S→N (-90 … 89.75).
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)
    if no_cds:
        print(f"  WARNING: {cache_path} not found and --no-cds set — filling with zeros")
        return np.zeros((720, 1440), dtype=np.float32)
    try:
        import cdsapi
        import xarray as xr
    except ImportError:
        print("  WARNING: cdsapi/xarray not available — filling with zeros")
        return np.zeros((720, 1440), dtype=np.float32)

    print(f"  Downloading ERA5 tp for {date_str} from CDS…")
    year, month, day = date_str.split("-")
    c = cdsapi.Client(quiet=True)
    with tempfile.TemporaryDirectory() as tmp:
        nc = os.path.join(tmp, "era5.nc")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable":     ["total_precipitation"],
            "year":  year, "month": month, "day": day,
            "time":  [f"{h:02d}:00" for h in range(24)],
            "format": "netcdf",
        }, nc)
        ds = xr.open_dataset(nc)
        tp_24h = ds["tp"].sum(dim="valid_time").values.astype(np.float32)
        # ERA5 lat is N→S; flip to S→N
        tp_24h = tp_24h[::-1, :]
    np.save(cache_path, tp_24h)
    print(f"  Cached → {cache_path}")
    return tp_24h


# ── Load ERA5 columns (only when --all-cols requested) ────────────────────────
if args.all_cols:
    print("Loading ERA5 initial (2020-01-01)…")
    era5_init_025 = download_era5_tp(
        "2020-01-01",
        os.path.join(R, "era5_tp_2020-01-01.npy"),
        args.no_cds,
    )
    era5_init_1   = era5_init_025[::4, ::4]

    print("Loading ERA5 truth (2020-01-02)…")
    era5_truth_025 = download_era5_tp(
        "2020-01-02",
        os.path.join(R, "era5_tp_2020-01-02.npy"),
        args.no_cds,
    )
    era5_truth_1   = era5_truth_025[::4, ::4]

# ── Load GenCast forecasts ────────────────────────────────────────────────────
print("Loading GenCast NPZ files…")
gc1   = np.load(os.path.join(R, "gencast-1.0_step1.npz"))
gc025 = np.load(os.path.join(R, "gencast-0.25_step1.npz"))
gc_tp_1   = gc1["total_precipitation_24hr"][0, 0]    # (180, 360)
gc_tp_025 = gc025["total_precipitation_24hr"][0, 0]  # (720, 1440)

# ── Load ORBIT-2 predictions ──────────────────────────────────────────────────
print("Loading ORBIT-2 predictions…")
preds_dir = os.path.join(R, "preds")
pred_1   = np.expm1(np.load(os.path.join(preds_dir, "gencast-1.0_step1_0_preds.npy")))  / 1000.0
pred_025 = np.expm1(np.load(os.path.join(preds_dir, "gencast-0.25_step1_0_preds.npy"))) / 1000.0
# Flip N-S to match S→N lat convention
pred_1   = pred_1[::-1]
pred_025 = pred_025[::-1]

# ── Grid coordinates ──────────────────────────────────────────────────────────
# 0.25° grid: -90 → 89.75
lat_025 = np.linspace(-90, 89.75, 720)
lon_025 = np.linspace(0, 359.75, 1440)
lon_025_plot = np.where(lon_025 > 180, lon_025 - 360, lon_025)
ext_025 = [lon_025_plot.min(), lon_025_plot.max(), lat_025.min(), lat_025.max()]

# 1° grid
lat_1 = lat_025[::4]
lon_1 = lon_025[::4]
lon_1_plot = np.where(lon_1 > 180, lon_1 - 360, lon_1)
ext_1 = [lon_1_plot.min(), lon_1_plot.max(), lat_1.min(), lat_1.max()]

# ORBIT-2 output is always 0.25° (upscaled from 1° input)
ext_pred_1   = ext_025
ext_pred_025 = ext_025   # same for now; adjust if different resolution

# ── Shared colour scale ───────────────────────────────────────────────────────
base_vals = np.concatenate([gc_tp_1.ravel(), gc_tp_025.ravel(),
                             pred_1.ravel(), pred_025.ravel()])
if args.all_cols:
    base_vals = np.concatenate([base_vals,
                                 era5_init_025.ravel(), era5_truth_025.ravel()])
vmin = 0.0
vmax = float(np.nanpercentile(base_vals[base_vals > 0], 99)) if (base_vals > 0).any() else 0.01

# ── Figure layout ─────────────────────────────────────────────────────────────
data_crs  = ccrs.PlateCarree()
n_cols    = 4 if args.all_cols else 2
fig_w     = 22 if args.all_cols else 12
col_titles_all = [
    "ERA5 initial\n(2020-01-01)",
    "GenCast forecast\n(2020-01-02)",
    "ERA5 ground truth\n(2020-01-02)",
    "ORBIT-2 downscaled\n(2020-01-02)",
]
col_titles = col_titles_all if args.all_cols else [col_titles_all[1], col_titles_all[3]]
row_labels = ["1.0° input", "0.25° input"]

fig = plt.figure(figsize=(fig_w, 9))
gs = gridspec.GridSpec(2, n_cols + 1, figure=fig,
                       width_ratios=[10] * n_cols + [0.4],
                       left=0.05, right=0.97,
                       top=0.93, bottom=0.06,
                       wspace=0.03, hspace=0.15)

# Build panel list for each mode
if args.all_cols:
    panels = [
        [(era5_init_1,    ext_1,        "1°"),
         (gc_tp_1,        ext_1,        "1°"),
         (era5_truth_1,   ext_1,        "1°"),
         (pred_1,         ext_pred_1,   "0.25°")],
        [(era5_init_025,  ext_025,      "0.25°"),
         (gc_tp_025,      ext_025,      "0.25°"),
         (era5_truth_025, ext_025,      "0.25°"),
         (pred_025,       ext_pred_025, "higher res")],
    ]
else:
    panels = [
        [(gc_tp_1,   ext_1,        "1°"),
         (pred_1,    ext_pred_1,   "0.25°")],
        [(gc_tp_025, ext_025,      "0.25°"),
         (pred_025,  ext_pred_025, "higher res")],
    ]

im = None
for row in range(2):
    for col in range(n_cols):
        ax = fig.add_subplot(gs[row, col], projection=data_crs)
        data, ext, res_label = panels[row][col]
        ax.set_extent([-180, 180, -90, 90], crs=data_crs)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
        im = ax.imshow(
            data[::-1],
            origin="upper",
            extent=ext,
            transform=data_crs,
            cmap=CMAP, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto", zorder=2,
        )
        if row == 0:
            ax.set_title(col_titles[col], fontsize=11, pad=4)
        if col == 0:
            ax.text(-0.12, 0.5, row_labels[row], transform=ax.transAxes,
                    fontsize=11, va="center", ha="right", rotation=90,
                    fontweight="bold")
        ax.text(0.02, 0.03, res_label, transform=ax.transAxes,
                fontsize=8, va="bottom", color="white",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1))

# Shared colorbar
cax = fig.add_subplot(gs[:, n_cols])
cb = fig.colorbar(im, cax=cax, label="Precipitation (m/day)")
cb.ax.tick_params(labelsize=9)
cb.ax.yaxis.label.set_size(10)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {args.out}")
