Help the user run ORBIT-2 precipitation downscaling inference.

Ask which input source they are using:
- **Option A — ERA5-IMERG**: standard benchmark inference
- **Option B — GenCast GRIB**: convert GenCast output first, then run inference

---

## Option A — ERA5-IMERG inference

Verify the data layout:
```
$OLCF_DATA_PATH/ERA5_IMERG_input/test/2021_0.npz   (180×360, 1° low-res)
$OLCF_DATA_PATH/ERA5_IMERG_output/test/2021_0.npz  (720×1440, 0.25° high-res)
$OLCF_DATA_PATH/ERA5_IMERG_output/lat.npy
$OLCF_DATA_PATH/ERA5_IMERG_output/lon.npy
```

Check whether `OLCF_DATA_PATH` is set:
```bash
echo $OLCF_DATA_PATH
```

Inside the Docker container, run:
```bash
bash run_infer.sh
```

Predictions are saved to `./results/` as `{stem}_{timestep}_preds.npy`.

---

## Option B — GenCast GRIB input

### Step 1 — Convert GRIB to NPZ

Confirm they have:
- A GenCast GRIB file (from `ai-models` or similar)
- CDS credentials at `~/.cdsapirc` (needed for ERA5 auxiliary variables)

Run the conversion:
```bash
python grib_to_npz.py \
    --grib data/gencast-1.0.grib \
    --valid-time 2020-01-02 \
    --date 2020-01-01 \
    --out results/gencast_step1.npz
```

If CDS is unavailable, add `--no-cds` (auxiliary variables will be zeroed).

### Step 2 — Run inference on the NPZ

Inside the Docker container:
```bash
bash run_gencast.sh
```

The script reads the converted NPZ and writes predictions to `./results/`.

---

## Checking results

After inference completes:
```bash
ls results/
# Expect: gencast_step1_0_preds.npy  (or era5_0_preds.npy for ERA5 path)
python -c "import numpy as np; a=np.load('results/gencast_step1_0_preds.npy'); print(a.shape, a.min(), a.max())"
```

Expected shape: `(1, 720, 1440)` — a single 0.25° global precipitation field.

---

## Common issues

| Symptom | Fix |
|---|---|
| `FileNotFoundError: normalize_mean.npz` | YAML config data path is wrong — run `/configure-orbit2` to fix |
| `CUDA/HIP device not found` | Container launched without GPU flags — check `docker_run_amd.sh` |
| `Killed` mid-inference | Docker memory limit too low — add `--memory=32g` to the docker run command |
| `total_precipitation_24hr` is NaN | Normal for some GenCast members — `grib_to_npz.py` handles this |
| `ModuleNotFoundError: utils` | Run `export PYTHONPATH=/workspace/ORBIT-2/examples:$PYTHONPATH` inside the container |

After resolving any issues, confirm the output `.npy` file exists and proceed to `/plot-results`.
