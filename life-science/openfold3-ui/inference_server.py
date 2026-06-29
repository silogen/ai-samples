"""Persistent OpenFold3 inference server.

Runs inside the OF3 Docker container. Loads the model once on startup
and keeps it in GPU memory. Accepts prediction requests via HTTP.
"""

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inference_server")

app = Flask(__name__)

RUNNER = None
MODEL_READY = False
JOBS = {}
JOB_LOCK = threading.Lock()
WORK_QUEUE = queue.Queue()

CACHED_PROPS_TO_INVALIDATE = [
    "output_dir",
    "log_dir",
    "callbacks",
    "trainer",
    "data_module_config",
    "lightning_data_module",
    "use_msa_server",
]


def load_model():
    global RUNNER, MODEL_READY

    from openfold3.core.config import config_utils
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.import_utils import _torch_gpu_setup
    from openfold3.entry_points.validator import InferenceExperimentConfig

    _torch_gpu_setup()

    runner_yaml = os.environ.get("RUNNER_YAML")
    runner_args = config_utils.load_yaml(Path(runner_yaml)) if runner_yaml else {}

    num_diffusion_samples = int(os.environ.get("NUM_DIFFUSION_SAMPLES", "1"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/results/pfui_jobs"))

    expt_config = InferenceExperimentConfig(**runner_args)
    RUNNER = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=num_diffusion_samples,
        output_dir=output_dir,
        use_msa_server=False,
        use_templates=False,
    )

    log.info("Loading model checkpoint...")
    RUNNER.setup()

    # Prevent PL teardown from moving the model off GPU after each predict call
    RUNNER.lightning_module.cpu = lambda *a, **kw: RUNNER.lightning_module

    MODEL_READY = True
    log.info("Model loaded and ready.")


def invalidate_runner_caches():
    for prop in CACHED_PROPS_TO_INVALIDATE:
        try:
            del RUNNER.__dict__[prop]
        except KeyError:
            pass


def worker_loop():
    """Processes jobs sequentially from the queue (single GPU)."""
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    while True:
        job_id, query_dict, output_dir, use_msa_server = WORK_QUEUE.get()
        query_dir = None
        try:
            with JOB_LOCK:
                JOBS[job_id]["status"] = "running"

            for colabfold_tmp in Path("/tmp").glob("**/colabfold_msas"):
                if colabfold_tmp.is_dir():
                    shutil.rmtree(colabfold_tmp, ignore_errors=True)

            query_dir = Path(tempfile.mkdtemp())
            query_path = query_dir / "query.json"
            query_path.write_text(json.dumps(query_dict))

            query_set = InferenceQuerySet.from_json(query_path)

            out_path = Path(output_dir)
            RUNNER.experiment_config.experiment_settings.output_dir = out_path
            RUNNER.experiment_config.experiment_settings.use_msa_server = use_msa_server
            invalidate_runner_caches()
            RUNNER.run(query_set)

            cif_files = list(out_path.rglob("*_model.cif"))
            if cif_files:
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "completed"
                log.info("Job %s completed", job_id)
            else:
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id]["error"] = "No output structure produced"
                log.warning("Job %s produced no CIF files", job_id)

        except Exception as e:
            log.error("Job %s failed: %s", job_id, traceback.format_exc())
            with JOB_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)
        finally:
            if query_dir and query_dir.exists():
                shutil.rmtree(query_dir, ignore_errors=True)
            WORK_QUEUE.task_done()


@app.route("/health")
def health():
    return jsonify({
        "model_ready": MODEL_READY,
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] == "running"),
        "total_jobs": len(JOBS),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded yet"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400
    for key in ("job_id", "query", "output_dir"):
        if key not in data:
            return jsonify({"error": f"Missing required field: {key}"}), 400

    job_id = data["job_id"]
    query = data["query"]
    output_dir = data["output_dir"]
    use_msa_server = data.get("use_msa_server", False)

    with JOB_LOCK:
        if job_id in JOBS:
            return jsonify({"error": f"Job {job_id} already exists"}), 409
        JOBS[job_id] = {"status": "queued"}

    WORK_QUEUE.put((job_id, query, output_dir, use_msa_server))

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/status/<job_id>")
def job_status(job_id):
    with JOB_LOCK:
        info = JOBS.get(job_id)
    if not info:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, **info})


if __name__ == "__main__":
    port = int(os.environ.get("INFERENCE_PORT", "8061"))
    load_model()
    threading.Thread(target=worker_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=port)
