Help the user adapt the ORBIT-2 setup scripts in this directory to their system.

The files to configure are:
- `setup.sh` — clones ORBIT-2 and builds the Docker image with `docker build`
- `Dockerfile` — builds the orbit2 image from the ROCm PyTorch base, installs all system and Python dependencies
- `docker_run_amd.sh` — launches container with AMD Container Toolkit (`--runtime=amd`)
- `docker_run_standard.sh` — launches container without AMD Container Toolkit (`--device=/dev/kfd`)
- `run_era5.sh` — runs downscaling inference on ERA5-IMERG data
- `run_gencast.sh` — runs downscaling inference on GenCast GRIB output
- `load_gencast.py` — loads GenCast GRIB output with cfgrib
- `config_patch.yaml` — ORBIT-2 data path config snippet

Ask the user these questions one at a time and update the relevant files accordingly:

1. **Docker runtime**: Do they have the AMD Container Toolkit installed?
   - Yes → use `docker_run_amd.sh` (already correct)
   - No → use `docker_run_standard.sh` (already correct), and note that `docker_run_amd.sh` is not needed

2. **GPU visibility**: How many GPUs do they want to expose?
   - If using AMD Container Toolkit: update `AMD_VISIBLE_DEVICES` in `docker_run_amd.sh` (e.g., `all`, `0`, `0,1`)
   - If using standard: update `--device=/dev/dri` to a specific device if needed

3. **Workspace path**: What local directory should be mounted as `/workspace/` in the container?
   - Default is `$(pwd)`. Replace `-v $(pwd):/workspace/` in both docker run scripts with their absolute path if they prefer a fixed mount.

4. **Dockerfile base image**: Do they need a different ROCm or PyTorch version?
   - Update the `FROM` line in `Dockerfile` and the corresponding `--index-url` pip flags to match their ROCm version.

5. **Checkpoint paths**: Where will they store the `.ckpt` and `.yaml` checkpoint files?
   - Update `--checkpoint` and the first positional argument (yaml path) in `run_era5.sh` and `run_gencast.sh`

6. **GenCast GRIB path**: Where is their `gencast-1.0.grib` output file?
   - Update `--gencast-output` in `run_gencast.sh`
   - Update the path string in `load_gencast.py`

7. **ERA5-IMERG data paths**: Where are their `ERA5_IMERG_input` and `ERA5_IMERG_output` data directories?
   - Update `low_res_dir` and `high_res_dir` in `config_patch.yaml`

8. **Inference index and date**: What sample index and GenCast forecast date do they want to run?
   - Update `--index` in `run_era5.sh` and `run_gencast.sh`
   - Update `--demo-time` in `run_gencast.sh`

After collecting answers, apply all changes to the relevant files using Edit. Summarize what was changed at the end.
