Help the user get started with this ORBIT-2 precipitation downscaling repository.

This repository contains tools to:
1. Run ORBIT-2 inference on ERA5 or GenCast input data
2. Convert GenCast GRIB files to the NPZ format ORBIT-2 expects
3. Evaluate and visualise downscaling results

Walk the user through the following steps in order, checking what they already have before proceeding.

---

## Step 1 — Prerequisites

Ask what environment they are working in:
- **Local ROCm machine**: they can build and run the Docker container directly
- **Remote HPC/cluster**: they may need to adapt the Docker or Singularity setup
- **Just the Python tools** (plot_comparison.py, grib_to_npz.py, etc.): they only need the conda environment

If they only need the Python tools (no inference), skip to Step 5.

---

## Step 2 — Docker setup

Direct them to run `/configure-orbit2` for interactive Docker configuration, or explain the quick path:

```bash
# Build the container
bash setup.sh

# Launch (AMD GPU)
bash docker_run_amd.sh

# Launch (standard Docker)
bash docker_run_standard.sh
```

Key files:
- `Dockerfile` — installs ROCm PyTorch + all Python deps (cfgrib, cartopy, cmocean, pyshtools, scikit-image)
- `docker_run_amd.sh` / `docker_run_standard.sh` — mounts workspace and exposes GPUs

---

## Step 3 — Checkpoints

The ORBIT-2 checkpoint (`.ckpt`) and config (`.yaml`) must be placed at:
```
./checkpoints/global-finetune/global_9.5m_precipitation.yaml
./checkpoints/global-finetune/global_9.5m_precipitation.ckpt
```
Available at: https://huggingface.co/jychoi-hpc/ORBIT-2

---

## Step 4 — Input data

**Option A — ERA5-IMERG (standard inference)**
Data must be in the directory structure:
```
$OLCF_DATA_PATH/ERA5_IMERG_input/test/2021_0.npz   (180×360, 1° low-res)
$OLCF_DATA_PATH/ERA5_IMERG_output/test/2021_0.npz  (720×1440, 0.25° high-res)
$OLCF_DATA_PATH/ERA5_IMERG_output/lat.npy
$OLCF_DATA_PATH/ERA5_IMERG_output/lon.npy
```
Set `OLCF_DATA_PATH` to point to the root data directory.

**Option B — GenCast GRIB input**
If they have a GenCast GRIB file (from ai-models or similar), convert it first:
```bash
python grib_to_npz.py \
    --grib data/gencast-1.0.grib \
    --valid-time 2020-01-02 \
    --date 2020-01-01 \
    --out results/gencast_step1.npz
```
This downloads missing ERA5 variables (landcover, soil moisture) from CDS automatically.
CDS credentials must be in `~/.cdsapirc`.

---

## Step 5 — Local Python tools (no Docker)

For the analysis and visualisation scripts, create the conda environment:
```bash
conda create -n vis-wp python=3.10
conda activate vis-wp
conda install -c conda-forge cartopy cmocean pyshtools scikit-image
pip install cfgrib numpy matplotlib cdsapi
```

Set the data path:
```bash
export OLCF_DATA_PATH=/path/to/data
```

---

## Step 6 — Run inference

**ERA5 input:**
```bash
# Inside the container
bash run_era5.sh
```

**GenCast input:**
```bash
# Inside the container
bash run_gencast.sh
```

Predictions are saved to `./results/` as `{stem}_{timestep}_preds.npy`.

---

## Step 7 — Visualise and evaluate

**Comparison plots** (comparison, differences, FFT):
```bash
conda run -n vis-wp python plot_comparison.py --plate-carree
conda run -n vis-wp python plot_comparison.py --fft --plate-carree
```

**GenCast vs ORBIT-2 figure** (2×2 default):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python plot_gencast.py
# Full 8-panel with ERA5 truth:
OLCF_DATA_PATH=... conda run -n vis-wp python plot_gencast.py --all-cols
```

**Spherical harmonic analysis:**
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python spherical_harmonics.py
```

**Metrics:**
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python evaluate_metrics.py --pred-dir ./results
```

---

## Step 8 — Advanced analysis

**High-pass filtered FFT** (isolates fine-scale structure above LR Nyquist):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python plot_hpfft.py
```

**Spherical harmonic power spectrum** (scale-by-scale energy analysis):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python spherical_harmonics.py --pred-dir ./results
```

**Compare GenCast NPZ vs ERA5** (check variable-by-variable alignment):
```bash
conda run -n vis-wp python compare_npz.py \
    --ref $OLCF_DATA_PATH/ERA5_IMERG_input/test/2021_0.npz --ref-idx 1 \
    --pred results/gencast_step1.npz
```

**Showcase downscaling figure** (for blog/presentation):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python downscaling_figure.py
```

**FFT/SH all at once:**
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python plot_comparison.py \
    --fft --plate-carree
OLCF_DATA_PATH=... conda run -n vis-wp python spherical_harmonics.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: utils` | Add `export PYTHONPATH=/workspace/ORBIT-2/examples:$PYTHONPATH` |
| `FileNotFoundError: normalize_mean.npz` | YAML config has wrong data path — update `ERA5-IMERG-FUSED` paths |
| `Killed` during inference | Docker container memory limit too low — increase `--memory` flag |
| CDS download hangs | ECMWF CDS maintenance — retry later; use `--no-cds` flag as fallback |
| `total_precipitation_24hr` is NaN | GenCast GRIB member/step NaN — `grib_to_npz.py` handles this automatically |

---

After understanding their situation, guide them to the relevant step and help them resolve any issues interactively.
