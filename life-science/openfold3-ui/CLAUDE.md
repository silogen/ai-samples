# CLAUDE.md

Guidance for Claude Code (and other agentic assistants) working in this repo.

## What this project is

PFUI is a local web UI for running OpenFold3 protein structure prediction on a
SLURM cluster with AMD MI300 GPUs. The UI is a Flask app served from a Docker
container running inside a SLURM job; users access it via SSH tunnel.

Two launch modes:

- **GPU mode** (`launch-gpu.slurm`): one SLURM job runs both the UI and a
  persistent inference server (`inference_server.py`) in separate containers.
  The OpenFold3 model is loaded **once** into GPU memory and reused across
  predictions. Jobs are queued and run sequentially on the single GPU.
- **CPU mode** (`launch-cpu.slurm`): the UI runs on a CPU-only SLURM job and
  spawns a new GPU SLURM job per prediction using
  `configs/OpenFold3/inference_template.slurm`. Slower per-job (cold-loads the
  model every time) but trivially parallelizable.

The UI itself is the same in both modes; behavior is selected via the
`INFERENCE_MODE` env var read by `app.py` (`server` or `slurm`).

## Repository layout

- `app.py` — Flask UI backend. Routes: `/`, `/api/generate`, `/api/status/<id>`,
  `/api/file/<id>/<path>`, `/api/jobs`, `/api/debug` (gated by `PFUI_DEBUG`).
- `inference_server.py` — Persistent inference server (GPU mode only). Runs
  inside the OF3 container. Loads model on startup, exposes `/health`,
  `/predict`, `/status/<id>`. Jobs are processed by a single worker thread
  consuming from a `queue.Queue` so concurrent requests serialize correctly on
  the one GPU.
- `templates/index.html` — Single-page frontend. Mol\* viewer on the right,
  multi-chain input on the left (Protein/RNA/DNA/Ligand), MSA mode selector
  (none / ColabFold / Upload), job list, JSON output toggle.
- `Dockerfile.server` — UI container image (`pfui:latest`). Small Python+Flask.
- `configs/OpenFold3/Dockerfile.rocm` — OpenFold3 inference image
  (`${DOCKER_IMAGE}` from `config.env`). Built fresh on every launch.
- `configs/OpenFold3/inference_template.slurm` — SLURM script template used in
  CPU mode only. `app.py` substitutes `{{...}}` placeholders before `sbatch`.
- `configs/OpenFold3/inference_config_pfui.yml` — OpenFold3 runner YAML
  (settings that aren't CLI flags, e.g. kernel choices, recycles).
- `launch-gpu.slurm`, `launch-cpu.slurm` — Top-level entry points.
- `config.env` — Centralized paths/ports/image names sourced by the launch
  scripts and passed into containers as `-e` env vars.
- `jobs/` — Per-job working dirs on the host (query.json, meta.json, uploaded
  MSAs, generated slurm script in CPU mode). Mounted into containers at
  `/jobs`. Slurm logs go in `jobs/slurm_logs/`.
- `${RESULTS_DIR}/pfui_jobs/<job_id>/` — Where OpenFold3 writes output CIFs and
  confidence JSONs. Mounted as `/results` in containers.

## Configuration

All host-side configuration lives in `config.env`. It's sourced by the launch
scripts and the relevant variables are forwarded into the UI container:

```
PFUI_DIR          # repo root on host (also hardcoded in launch scripts)
RESULTS_DIR       # where OF3 outputs go
CACHE_DIR         # HuggingFace / OpenFold weight cache (use fast local storage)
OF3_DOCKERFILE    # path to OF3 Dockerfile (built every launch)
DOCKER_IMAGE      # tag for OF3 image
PFUI_PORT         # UI port (8060)
INFERENCE_PORT    # GPU-mode inference server port (8061, internal only)
```

Environment-specific, set in the SLURM scripts:
- `#SBATCH --partition=CHANGE_ME` — must be edited to a real partition.
- `#SBATCH --gres=gpu:mi300:1` (GPU jobs only) — adjust to your cluster's gres.
- `PFUI_DIR="${SLURM_SUBMIT_DIR}"` — the dir you ran `sbatch` from. We use this
  instead of `${BASH_SOURCE[0]}` because the latter resolves to SLURM's spool
  dir inside a running job, not the script location. `config.env` keeps a
  `${PFUI_DIR:-$HOME/pfui}` fallback for docker compose / bare-metal runs.

## Key data flow

### GPU mode (`INFERENCE_MODE=server`)
1. User submits chains + MSA mode from the browser.
2. `app.py` validates input, assigns chain IDs (A, B, C, ...), writes
   `jobs/<id>/query.json` and `meta.json`, then `POST`s
   `{job_id, query, output_dir, use_msa_server}` to
   `http://localhost:8061/predict`. Returns immediately (202 Accepted).
3. `inference_server.py` enqueues the job. The worker thread:
   - Wipes any stale `**/colabfold_msas` dirs under `/tmp` (avoids the
     `seq_to_rep_id.json` `KeyError` bug across runs).
   - Writes the query JSON to a fresh `tempfile.mkdtemp()` dir.
   - Mutates `RUNNER.experiment_config.experiment_settings.{output_dir,
     use_msa_server}` and calls `invalidate_runner_caches()` to drop cached
     properties (`trainer`, `lightning_data_module`, etc.) so PL re-reads the
     new settings.
   - Calls `RUNNER.run(query_set)`.
   - **Post-run check**: if no `*_model.cif` files exist in the output dir, the
     job is marked `failed` — OpenFold sometimes swallows errors internally.
4. Frontend polls `/api/status/<id>`. `app.py` first checks for output CIFs
   (definitive), then queries the inference server for in-flight status.

### CPU mode (`INFERENCE_MODE=slurm`)
Same UI flow, but `app.py` substitutes `{{...}}` placeholders in
`inference_template.slurm` and calls `sbatch`. Status is determined via
`squeue -j <slurm_job_id>` — if the job is gone and no CIFs exist, it's marked
`failed`.

## Gotchas / non-obvious details

- **PL teardown moves model to CPU.** Without intervention, after each
  `Trainer.predict()` call PyTorch Lightning calls `model.cpu()` during
  teardown, which crashes on subsequent calls with HIP errors. We patch it out
  in `load_model()`:
  ```python
  RUNNER.lightning_module.cpu = lambda *a, **kw: RUNNER.lightning_module
  ```
- **ColabFold mapping files persist between jobs** in
  `/tmp/.../colabfold_msas/mappings/`. If a previous job used uploaded MSAs and
  the next uses ColabFold (or different chain counts), stale mappings cause
  `KeyError` during MSA featurization. The worker wipes them at the start of
  every job using a `**/colabfold_msas` glob, so the cleanup survives upstream
  path changes (it has changed once already).
- **Sequence sanitization.** `assign_chain_ids()` strips *all* whitespace from
  sequences (`"".join(seq.split())`). Newlines pasted into the UI textareas
  used to be interpreted as unknown residues and broke featurization.
- **User mapping in Docker.** Both UI and inference containers run as
  `$(id -u):$(id -g)` with `/etc/passwd` and `/etc/group` bind-mounted in.
  SLURM tools won't work otherwise (`Invalid user for SlurmUser slurm`).
- **`alpine chown` pre-step** in the launch scripts is required so the
  unprivileged container user can write to `${RESULTS_DIR}/pfui_jobs/` and
  `jobs/slurm_logs/`, and so the user can later delete those files from the
  host.
- **`set -e` + `curl` health check.** The health-check `curl` in
  `launch-gpu.slurm` is followed by `|| true` to keep `set -euo pipefail` from
  aborting the script on the expected "connection refused" before the server
  binds.
- **`use_triton_triangle_kernels` vs `use_cueq_triangle_kernels`.** Upstream
  OpenFold renamed kernel flags; the YAML must match the version of the OF3
  source being built. If the inference server crashes immediately on startup
  with `KeyError: 'Key "..." does not exist'`, this is the cause.
- **`/opt/venv/bin/python`, not `python`.** OpenFold3 lives in a venv inside
  the inference image. Always invoke it via `/opt/venv/bin/...`; bare `python`
  is the system Python and won't see `openfold3`.

## How to run

```sh
sbatch launch-gpu.slurm   # recommended
# or
sbatch launch-cpu.slurm
```

Watch the slurm `.out` for the SSH tunnel command, then in another terminal:
```sh
ssh -N -L 8060:<node>:8060 <user>@<gateway>
```
Open <http://localhost:8060>.

## Coding conventions for this repo

- Be concise. Don't add try/except just to "be safe" — the user explicitly
  rejects defensive wrapping unless an error path is actually meaningful.
- Don't add narrating comments (`# loop over chains`). Only comment to explain
  *why* something non-obvious is being done (e.g. the `cpu = lambda ...` patch).
- Don't add data-availability checks unless requested.
- Prefer editing existing files. Don't create new files unless the user asks
  for one.
- When changing `inference_server.py`: it's mounted as a read-only volume into
  the container, so changes take effect on next SLURM job restart — **no
  Docker rebuild needed**. `app.py` and `templates/index.html` are baked into
  `pfui:latest`, so they require rebuilding the UI image (which the launch
  scripts do automatically every launch).

## When the user asks for "the inference server" vs "the UI"

- "UI" / "frontend" / `pfui:latest` → `app.py`, `templates/index.html`,
  `Dockerfile.server`.
- "Inference server" / "GPU server" → `inference_server.py`, runs inside the
  OF3 image (`${DOCKER_IMAGE}`).
- "OpenFold" / "OF3" / "the model" → upstream OpenFold3 code, *not* in this
  repo. Source lives elsewhere (see `OF3_DOCKERFILE` in `config.env`). Don't
  edit OpenFold source from here unless explicitly told to.
