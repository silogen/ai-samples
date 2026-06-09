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
5×3 panel comparison figures for all timesteps:
  - comparison:  low-res upscaled | high-res ground truth | ORBIT-2 prediction
  - differences: ground truth − low-res | downscaled − low-res | downscaled − ground truth
  - fft:         FFT magnitude of the three differences

Panels use cartopy PlateCarree projection with lat/lon from the high-res grid.
"""

import argparse
import cmocean
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _plotly_available = True
except ImportError:
    _plotly_available = False


def save_globe_html(diff_hr_lr, diff_pred_lr, diff_pred_hr, lat, lon, abs_max,
                    output_path, stride=4):
    """Three synchronized interactive globes saved as an HTML file.

    Camera sync is handled entirely in JavaScript via a post_script so the file
    is embeddable in Markdown / HTML with no Python kernel required.
    Plotly is loaded from CDN (include_plotlyjs="cdn") to keep file size small
    (~3 MB smaller than self-contained); network access is required to render.
    """
    if not _plotly_available:
        print("plotly not installed — skipping globe HTML export")
        return

    lat_r = np.deg2rad(lat[::stride])
    lon_r = np.deg2rad(lon[::stride])
    LON_s, LAT_s = np.meshgrid(lon_r, lat_r)
    X = np.cos(LAT_s) * np.cos(LON_s)
    Y = np.cos(LAT_s) * np.sin(LON_s)
    Z = np.sin(LAT_s)

    # Coastline segments projected onto the unit sphere
    def _coast_traces():
        traces = []
        for geom in cfeature.COASTLINE.geometries():
            parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for part in parts:
                coords = np.array(part.coords)
                lons_c = np.deg2rad(coords[:, 0])
                lats_c = np.deg2rad(coords[:, 1])
                traces.append(go.Scatter3d(
                    x=np.cos(lats_c) * np.cos(lons_c),
                    y=np.cos(lats_c) * np.sin(lons_c),
                    z=np.sin(lats_c),
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))
        return traces

    coast = _coast_traces()
    titles = ["Ground truth − Low-res", "Downscaled − Low-res", "Downscaled − Ground truth"]
    arrays = [diff_hr_lr[::stride, ::stride],
              diff_pred_lr[::stride, ::stride],
              diff_pred_hr[::stride, ::stride]]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=titles,
        horizontal_spacing=0.02,
    )

    for col, (data, _) in enumerate(zip(arrays, titles), start=1):
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, surfacecolor=data,
                       colorscale="RdBu_r", cmin=-abs_max, cmax=abs_max,
                       showscale=(col == 3),
                       colorbar=dict(title=dict(text="m/day", side="right"),
                                     x=1.01, len=0.8) if col == 3 else None),
            row=1, col=col,
        )
        for tr in coast:
            fig.add_trace(tr, row=1, col=col)

    scene_kw = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor="white", aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=0.5, z=0.5)),
    )
    fig.update_layout(
        scene=scene_kw, scene2=scene_kw, scene3=scene_kw,
        width=1300, height=520,
        margin=dict(l=0, r=80, t=40, b=0),
    )

    # JavaScript: detect which scene the user dragged and mirror camera to the others
    js_sync = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    var scenes = ['scene', 'scene2', 'scene3'];
    var syncing = false;
    gd.on('plotly_relayout', function(ed) {
        if (syncing) return;
        var moved = null;
        Object.keys(ed).forEach(function(k) {
            var m = k.match(/^(scene\d*)\.camera/);
            if (m) moved = m[1];
        });
        if (!moved) return;
        var cam = gd.layout[moved].camera;
        syncing = true;
        var update = {};
        scenes.forEach(function(s) { if (s !== moved) update[s + '.camera'] = cam; });
        Plotly.relayout(gd, update).then(function() { syncing = false; });
    });
})();
"""
    fig.write_html(output_path, include_plotlyjs="cdn", post_script=js_sync, full_html=True)
    print(f"Saved {output_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--no-flip", action="store_true",
                    help="Do not flip prediction along N-S axis (default is to flip)")
parser.add_argument("--plate-carree", action="store_true",
                    help="Use PlateCarree projection + imshow (faster; default is Mollweide + pcolormesh)")
parser.add_argument("--pred-dir", default="./results",
                    help="Directory containing *_preds.npy files (default: ./results)")
parser.add_argument("--out-dir", default=None,
                    help="Output directory for plots (default: ./results or ./results_noflip)")
parser.add_argument("--globe", action="store_true",
                    help="Also render interactive 3-D globe HTML files (slow)")
parser.add_argument("--fft", action="store_true",
                    help="Also render FFT-of-differences figure")
parser.add_argument("--diff-only", action="store_true",
                    help="Only render the differences figure (skip comparison and FFT)")
parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"],
                    help="Output format for figures (default: png). svg/pdf disable rasterization.")
parser.add_argument("--scale", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                    metavar=("S_CMP", "S_DIFF", "S_FFT"),
                    help="Scale factors applied to the percentile max for comparison, differences, "
                         "and FFT colormaps (default: 1 1 1)")
parser.add_argument("--percentile", type=float, default=95.0,
                    help="Percentile used to compute vmax/abs_max (default: 95)")
args = parser.parse_args()

# Override with: export OLCF_DATA_PATH=/path/to/OLCF-data/Data
OLCF_data_path = os.environ.get("OLCF_DATA_PATH", "/workspace/Data")
low_res_dir    = os.path.join(OLCF_data_path, "ERA5_IMERG_input/test")
high_res_dir   = os.path.join(OLCF_data_path, "ERA5_IMERG_output/test")
pred_dir       = args.pred_dir
downscaled_dir = args.out_dir or (pred_dir + "_noflip" if args.no_flip else "./results")
npz_file       = "2021_0.npz"
variable       = "total_precipitation_24hr"
n_timesteps    = 5

# High-res lat/lon grid (1-D, shape 720 and 1440)
lat = np.load(os.path.join(OLCF_data_path, "ERA5_IMERG_output/lat.npy"))  # -90 .. 89.75
lon = np.load(os.path.join(OLCF_data_path, "ERA5_IMERG_output/lon.npy"))  # 0 .. 359.75
# cartopy PlateCarree uses -180..180 for longitude
lon_plot = np.where(lon > 180, lon - 360, lon)
LON, LAT = np.meshgrid(lon_plot, lat)

import datetime
_hr_npz = np.load(os.path.join(high_res_dir, npz_file))
lr_all  = np.load(os.path.join(low_res_dir, npz_file))[variable][:, 0, :, :]
hr_all  = _hr_npz[variable][:, 0, :, :]
_year   = int(os.path.splitext(npz_file)[0].split("_")[0])
_doys   = _hr_npz["days_of_year"][:, 0, 0, 0].astype(int)
timestamps = [
    (datetime.date(_year, 1, 1) + datetime.timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    for d in _doys
]

os.makedirs(downscaled_dir, exist_ok=True)

panels, diffs, ffts_power, ffts_phase = [], [], [], []
for i in range(n_timesteps):
    lr_up = np.repeat(np.repeat(lr_all[i], 4, axis=0), 4, axis=1)
    hr    = hr_all[i]
    stem  = os.path.splitext(npz_file)[0]
    pred_path = os.path.join(pred_dir, f"{stem}_{i}_preds.npy")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pred_dir, f"{i}_preds.npy")
    pred  = np.expm1(np.load(pred_path)) / 1000.0
    if not args.no_flip:
        pred = pred[::-1, :]

    panels.append((lr_up, hr, pred))

    d_hr_lr   = hr - lr_up
    d_pred_lr = pred - lr_up
    d_pred_hr = pred - hr
    diffs.append((d_hr_lr, d_pred_lr, d_pred_hr))

    if args.fft:
        def fft2d(arr):
            return np.fft.fftshift(np.fft.fft2(arr))
        _ffts = [fft2d(d) for d in (d_hr_lr, d_pred_lr, d_pred_hr)]
        ffts_power.append(tuple(np.abs(f)**2 for f in _ffts))
        ffts_phase.append(tuple(np.angle(f) for f in _ffts))

data_crs = ccrs.PlateCarree()
proj     = data_crs if args.plate_carree else ccrs.Mollweide()
# imshow extent: [lon_min, lon_max, lat_min, lat_max] in PlateCarree coords
_extent  = [lon_plot.min(), lon_plot.max(), lat.min(), lat.max()]

_rasterized = (args.format == "png")

def _render_geo(ax, data, cmap, vmin, vmax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    if args.plate_carree:
        ax.set_extent([-180, 180, -90, 90], crs=data_crs)
        return ax.imshow(data, origin="upper", extent=_extent, cmap=cmap,
                         vmin=vmin, vmax=vmax, transform=data_crs,
                         interpolation="nearest", aspect="auto")
    else:
        ax.set_global()
        return ax.pcolormesh(LON, LAT, data, cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=data_crs, rasterized=_rasterized)

if not args.diff_only:
    # --- Figure 1: comparison ---
    vmin = min(d.min() for row in panels for d in row)
    vmax = np.percentile(np.concatenate([d.ravel() for row in panels for d in row]),
                         99.0) * args.scale[0]
    titles = ["Low-res (upscaled)", "High-res ground truth", "ORBIT-2 downscaled"]

    fig, axes = plt.subplots(n_timesteps, 3, figsize=(18, 4 * n_timesteps),
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(hspace=0.0, wspace=0.02, right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    for row, (lr_up, hr, pred) in enumerate(panels):
        for col, data in enumerate([lr_up, hr, pred]):
            ax = axes[row, col]
            im = _render_geo(ax, data, "viridis", vmin, vmax)
            if row == 0:
                ax.set_title(titles[col], fontsize=16)
            if col == 0:
                ax.text(-0.08, 0.5, timestamps[row], transform=ax.transAxes,
                        fontsize=16, va="center", ha="right", rotation=90)
    fig.colorbar(im, cax=cax, label="m/day").ax.tick_params(labelsize=14)
    cax.yaxis.label.set_size(16)
    fname = f"comparison_all.{args.format}"
    plt.savefig(os.path.join(downscaled_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# --- RMS per timestep ---
_rms_header = f"{'t':>3}  {'GT−LR':>12}  {'Pred−LR':>12}  {'Pred−GT':>12}"
_rms_sep    = "-" * 45
_rms_rows   = []
rms_per_ts  = []
for i, (d1, d2, d3) in enumerate(diffs):
    rms = [np.sqrt(np.mean(d**2)) for d in (d1, d2, d3)]
    rms_per_ts.append(rms)
    _rms_rows.append(f"{i:>3}  {rms[0]:>12.6f}  {rms[1]:>12.6f}  {rms[2]:>12.6f}")
_rms_table = "\n".join([_rms_header, _rms_sep] + _rms_rows)
print(f"\n{_rms_table}\n")
rms_path = os.path.join(downscaled_dir, "rms.txt")
with open(rms_path, "w") as _f:
    _f.write(_rms_table + "\n")
print(f"Saved {rms_path}")

# --- Figure 2: differences (diverging colormap, symmetric around 0) ---
abs_max = np.percentile(np.concatenate([np.abs(d).ravel() for row in diffs for d in row]),
                        args.percentile) * args.scale[1]
titles_diff = ["Ground truth − Low-res", "Downscaled − Low-res", "Downscaled − Ground truth"]

fig, axes = plt.subplots(n_timesteps, 3, figsize=(18, 4 * n_timesteps),
                         subplot_kw={"projection": proj})
fig.subplots_adjust(hspace=0.0, wspace=0.02, right=0.88, bottom=0.12)
cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
for row, (d1, d2, d3) in enumerate(diffs):
    rms = rms_per_ts[row]
    for col, data in enumerate([d1, d2, d3]):
        ax = axes[row, col]
        im = _render_geo(ax, data, cmocean.cm.balance, -abs_max, abs_max)
        if row == 0:
            ax.set_title(titles_diff[col], fontsize=16)
        if col == 0:
            ax.text(-0.08, 0.5, timestamps[row], transform=ax.transAxes,
                    fontsize=16, va="center", ha="right", rotation=90)
        ax.text(0.5, -0.15, f"RMS={rms[col]:.4f}", transform=ax.transAxes,
                fontsize=14, va="bottom", ha="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
fig.colorbar(im, cax=cax, label="m/day").ax.tick_params(labelsize=14)
cax.yaxis.label.set_size(16)
fname = f"differences_all.{args.format}"
plt.savefig(os.path.join(downscaled_dir, fname), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fname}")

if args.fft:
    h, w = ffts_power[0][0].shape
    cy, cx = h // 2, w // 2
    ay, ax_ = h // 4, w // 4
    titles_fft = ["Ground truth − Low-res", "Downscaled − Low-res", "Downscaled − Ground truth"]

    # --- Figure 3a: Power spectrum ---
    vmax_pwr = np.percentile(np.concatenate([d.ravel() for row in ffts_power for d in row]),
                             args.percentile) * args.scale[2]
    fig, axes = plt.subplots(n_timesteps, 3, figsize=(18, 4 * n_timesteps))
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    for row, (f1, f2, f3) in enumerate(ffts_power):
        for col, data in enumerate([f1, f2, f3]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax_pwr, interpolation="nearest")
            if row == 0:
                ax.set_title(titles_fft[col], fontsize=16)
            ax.axis("off")
            if col == 0:
                ax.text(-0.08, 0.5, timestamps[row], transform=ax.transAxes,
                        fontsize=16, va="center", ha="right", rotation=90)
            if col < 2:
                line_kw = dict(color="white", linestyle="--", linewidth=1.5, alpha=0.8)
                for y in [cy - ay, cy + ay]:
                    ax.axhline(y, **line_kw)
                for x in [cx - ax_, cx + ax_]:
                    ax.axvline(x, **line_kw)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|FFT|²", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    fig.subplots_adjust(hspace=-0.4, wspace=0.02, right=0.88)
    plt.savefig(os.path.join(downscaled_dir, "fft_power.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fft_power.png")

    # --- Figure 3b: Phase ---
    fig, axes = plt.subplots(n_timesteps, 3, figsize=(18, 4 * n_timesteps))
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    for row, (f1, f2, f3) in enumerate(ffts_phase):
        for col, data in enumerate([f1, f2, f3]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                           interpolation="nearest")
            if row == 0:
                ax.set_title(titles_fft[col], fontsize=16)
            ax.axis("off")
            if col == 0:
                ax.text(-0.08, 0.5, timestamps[row], transform=ax.transAxes,
                        fontsize=16, va="center", ha="right", rotation=90)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Phase (rad)", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    fig.subplots_adjust(hspace=-0.4, wspace=0.02, right=0.88)
    plt.savefig(os.path.join(downscaled_dir, "fft_phase.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fft_phase.png")

if args.globe:
    stem = os.path.splitext(npz_file)[0]
    for i, (d_hr_lr, d_pred_lr, d_pred_hr) in enumerate(diffs):
        save_globe_html(
            d_hr_lr, d_pred_lr, d_pred_hr,
            lat, lon, abs_max,
            output_path=os.path.join(downscaled_dir, f"globe_{stem}_{i}.html"),
        )
