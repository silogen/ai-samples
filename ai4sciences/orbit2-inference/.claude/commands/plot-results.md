Help the user visualise and evaluate ORBIT-2 downscaling results.

First check that inference has been run and results exist:
```bash
ls results/
```
Expect `.npy` files such as `gencast_step1_0_preds.npy` or `era5_0_preds.npy`.

Also confirm the conda environment exists:
```bash
conda env list | grep vis-wp
```
If missing, create it:
```bash
conda create -n vis-wp python=3.10
conda activate vis-wp
conda install -c conda-forge cartopy cmocean pyshtools scikit-image
pip install cfgrib numpy matplotlib cdsapi
```

---

## Comparison plots (side-by-side + difference maps)

```bash
# Plate Carrée projection
OLCF_DATA_PATH=... conda run -n vis-wp python plot_comparison.py --plate-carree

# With FFT power spectrum panel
OLCF_DATA_PATH=... conda run -n vis-wp python plot_comparison.py --fft --plate-carree
```

Produces figures comparing low-res input, ORBIT-2 output, and high-res reference.

---

## GenCast vs ORBIT-2 figure

```bash
# 2×2 panel (default): GenCast LR, ORBIT-2 HR, IMERG reference, difference
OLCF_DATA_PATH=... conda run -n vis-wp python plot_gencast.py

# Full 8-panel including ERA5 truth columns
OLCF_DATA_PATH=... conda run -n vis-wp python plot_gencast.py --all-cols
```

---

## Downscaling showcase figure (for blog/presentation)

```bash
OLCF_DATA_PATH=... conda run -n vis-wp python downscaling_figure.py
```

---

## Quantitative metrics (RMSE, SSIM, bias)

```bash
OLCF_DATA_PATH=... conda run -n vis-wp python evaluate_metrics.py --pred-dir ./results
```

Prints a table of metrics comparing ORBIT-2 predictions to the IMERG reference.

---

## Spectral analysis

**FFT power spectrum** (spatial frequency content):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python plot_comparison.py --fft --plate-carree
```

**High-pass filtered FFT** (fine-scale structure above low-res Nyquist only):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python plot_hpfft.py
```

**Spherical harmonic power spectrum** (scale-by-scale energy):
```bash
OLCF_DATA_PATH=... conda run -n vis-wp python spherical_harmonics.py --pred-dir ./results
```

---

## GenCast NPZ variable check

If you want to verify GenCast variables align with ERA5 before running inference:
```bash
conda run -n vis-wp python compare_npz.py \
    --ref $OLCF_DATA_PATH/ERA5_IMERG_input/test/2021_0.npz --ref-idx 1 \
    --pred results/gencast_step1.npz
```

---

## Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cartopy` | Activate or recreate the `vis-wp` conda environment |
| `FileNotFoundError` for `.npy` | Run inference first with `/run-inference` |
| Blank or all-zero plot | Check `OLCF_DATA_PATH` is set and points to the right directory |
| `KeyError` in plot script | NPZ variable names differ — run `compare_npz.py` to inspect |

Ask the user which figure they want to produce and guide them step by step.
