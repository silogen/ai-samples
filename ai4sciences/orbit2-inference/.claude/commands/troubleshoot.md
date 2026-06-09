Help the user diagnose and fix problems with the ORBIT-2 inference pipeline.

Ask the user to describe the error or symptom they are seeing, then work through the relevant section below.

---

## Python import errors

**`ModuleNotFoundError: utils`**
The ORBIT-2 examples directory is not on the Python path.
```bash
export PYTHONPATH=/workspace/ORBIT-2/examples:$PYTHONPATH
```
Add this to the docker run script or your shell profile to make it permanent.

**`ModuleNotFoundError: cartopy` / `cmocean` / `cfgrib`**
The `vis-wp` conda environment is missing or incomplete.
```bash
conda create -n vis-wp python=3.10
conda activate vis-wp
conda install -c conda-forge cartopy cmocean pyshtools scikit-image
pip install cfgrib numpy matplotlib cdsapi
```

---

## Checkpoint / config errors

**`FileNotFoundError: normalize_mean.npz`** or similar data path errors
The YAML config points to wrong data directories. Check and fix:
```bash
cat checkpoints/global-finetune/global_9.5m_precipitation.yaml | grep -A5 ERA5
```
Update `low_res_dir` and `high_res_dir` to match your actual `$OLCF_DATA_PATH`. Run `/configure-orbit2` to do this interactively.

**`FileNotFoundError: .ckpt`**
Checkpoint not downloaded. Get it from: https://huggingface.co/jychoi-hpc/ORBIT-2
Place at: `./checkpoints/global-finetune/global_9.5m_precipitation.ckpt`

---

## GPU / Docker errors

**`CUDA/HIP device not found` or `No GPU available`**
The container was launched without GPU flags. Verify:
```bash
# AMD GPU
docker run --runtime=amd ...        # AMD Container Toolkit path
# or
docker run --device=/dev/kfd --device=/dev/dri ...   # manual device path
```
Run `/configure-orbit2` to fix the launch script.

**Process killed (`Killed`) during inference**
Docker memory limit is too low. Increase it:
```bash
docker run --memory=32g ...
```
Edit `docker_run_amd.sh` or `docker_run_standard.sh` to add this flag.

---

## Data / input errors

**`total_precipitation_24hr` is NaN in GenCast GRIB**
This is handled automatically by `grib_to_npz.py`. If you see it in a raw GRIB inspection, it is normal for some ensemble members.

**`KeyError` when loading NPZ**
Variable names in the NPZ don't match what the model expects. Inspect both:
```bash
python -c "import numpy as np; d=np.load('results/gencast_step1.npz'); print(list(d.keys()))"
python -c "import numpy as np; d=np.load('$OLCF_DATA_PATH/ERA5_IMERG_input/test/2021_0.npz'); print(list(d.keys()))"
```
Then run `compare_npz.py` for a full variable-by-variable diff.

**CDS download hangs or times out**
ECMWF CDS may be under maintenance. Options:
- Retry later
- Add `--no-cds` flag to `grib_to_npz.py` (auxiliary ERA5 variables will be zeroed)
- Check CDS status at https://cds.climate.copernicus.eu

---

## Output sanity checks

After inference, quickly verify the output:
```bash
python -c "
import numpy as np, glob
for f in sorted(glob.glob('results/*_preds.npy')):
    a = np.load(f)
    print(f, a.shape, f'min={a.min():.4f} max={a.max():.4f} nan={np.isnan(a).sum()}')
"
```
Expected: shape `(1, 720, 1440)`, no NaNs, values roughly in mm/day range (0–100).

---

If none of the above matches your error, paste the full traceback and I will help diagnose it.
