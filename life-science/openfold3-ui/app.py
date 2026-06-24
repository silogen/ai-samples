import json
import logging
import os
import string
import subprocess
import uuid
from pathlib import Path

import requests as http_requests
from flask import Flask, jsonify, render_template, request, send_file

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
log = logging.getLogger(__name__)

app = Flask(__name__)

JOBS_DIR = Path("/jobs")
RESULTS_BASE = Path("/results/pfui_jobs")
SLURM_LOGS_DIR = Path("/jobs/slurm_logs")
INFERENCE_TEMPLATE = Path("/app/configs/OpenFold3/inference_template.slurm")

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "slurm")
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8061")
DEBUG_ENABLED = os.environ.get("PFUI_DEBUG", "").lower() in ("1", "true", "yes")

TEMPLATE_VARS = {}
if INFERENCE_MODE == "slurm":
    _missing = [k for k in ["PFUI_DIR", "RESULTS_DIR", "CACHE_DIR", "DOCKER_IMAGE"] if k not in os.environ]
    if _missing:
        raise RuntimeError(f"Missing required env vars for slurm mode: {', '.join(_missing)}")
    TEMPLATE_VARS = {
        key: os.environ[key]
        for key in ["PFUI_DIR", "RESULTS_DIR", "CACHE_DIR", "DOCKER_IMAGE"]
    }

MAX_CHAINS = 26


@app.route("/")
def index():
    return render_template("index.html")


def assign_chain_ids(chains_input):
    """Convert chains with copies into OpenFold chain entries with assigned IDs."""
    chain_letters = list(string.ascii_uppercase)
    idx = 0
    of_chains = []
    for entry in chains_input:
        copies = entry.get("copies", 1)
        if idx + copies > MAX_CHAINS:
            raise ValueError(f"Total chain count exceeds {MAX_CHAINS} (A-Z)")
        ids = chain_letters[idx:idx + copies]
        idx += copies
        mol_type = entry.get("molecule_type", "protein")
        if mol_type in ("ligand", "ligand_ccd"):
            chain = {"molecule_type": "ligand", "chain_ids": ids}
            if mol_type == "ligand":
                if "smiles" not in entry or not entry["smiles"].strip():
                    raise ValueError("Ligand chain missing SMILES string")
                chain["smiles"] = entry["smiles"].strip()
            else:
                if "ccd" not in entry or not entry["ccd"].strip():
                    raise ValueError("Ligand chain missing CCD code")
                chain["ccd_codes"] = [c.strip().upper() for c in entry["ccd"].split(",") if c.strip()]
        else:
            chain = {"molecule_type": mol_type, "chain_ids": ids}
            if "sequence" not in entry or not entry["sequence"].strip():
                raise ValueError(f"{mol_type} chain missing sequence")
            chain["sequence"] = "".join(entry["sequence"].split())
        of_chains.append(chain)
    return of_chains


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        chains_input = json.loads(request.form.get("chains", "[]"))
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid chains JSON: {e}"}), 400

    msa_mode = request.form.get("msa_mode", "none")
    if not chains_input:
        return jsonify({"error": "No chains provided"}), 400

    try:
        of_chains = assign_chain_ids(chains_input)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    SLURM_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = RESULTS_BASE / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if msa_mode == "upload":
        msa_dir = job_dir / "msas"
        msa_dir.mkdir(exist_ok=True)
        for i, chain in enumerate(of_chains):
            if chain["molecule_type"] in ("protein", "rna"):
                file_key = f"msa_file_{i}"
                if file_key in request.files:
                    f = request.files[file_key]
                    ext = Path(f.filename).suffix or ".a3m"
                    filename = f"chain_{i}{ext}"
                    f.save(str(msa_dir / filename))
                    if INFERENCE_MODE == "server":
                        msa_path = f"/jobs/{job_id}/msas/{filename}"
                    else:
                        msa_path = f"/query/msas/{filename}"
                    chain["main_msa_file_paths"] = [msa_path]

    def _chain_summary(c):
        val = c.get("sequence") or c.get("smiles") or ",".join(c.get("ccd_codes", []))
        tag = {"protein": "P", "rna": "R", "dna": "D", "ligand": "L"}.get(c["molecule_type"], "?")
        return f"[{tag}] {val[:20]}...(x{len(c['chain_ids'])})"
    summary = " | ".join(_chain_summary(c) for c in of_chains)

    user_query = {"chains": of_chains}
    if msa_mode == "none":
        user_query["use_msas"] = False

    query = {"queries": {"user_query": user_query}}
    query_path = job_dir / "query.json"
    query_path.write_text(json.dumps(query, indent=2))

    use_msa_server = msa_mode == "colabfold"

    if INFERENCE_MODE == "server":
        try:
            resp = http_requests.post(
                f"{INFERENCE_SERVER_URL}/predict",
                json={
                    "job_id": job_id,
                    "query": query,
                    "output_dir": str(RESULTS_BASE / job_id),
                    "use_msa_server": use_msa_server,
                },
                timeout=10,
            )
        except http_requests.RequestException as e:
            return jsonify({"error": f"Inference server unreachable: {e}"}), 502
        if resp.status_code != 202:
            return jsonify({"error": f"Inference server error: {resp.text}"}), 500

        meta = {"job_id": job_id, "slurm_job_id": "gpu-server", "sequence": summary}
        (job_dir / "meta.json").write_text(json.dumps(meta))
        return jsonify({"job_id": job_id, "slurm_job_id": "gpu-server"})

    template = INFERENCE_TEMPLATE.read_text()
    replacements = {
        **TEMPLATE_VARS,
        "JOB_ID": job_id,
        "USE_MSA_SERVER": str(use_msa_server),
    }
    slurm_script = template
    for key, val in replacements.items():
        slurm_script = slurm_script.replace(f"{{{{{key}}}}}", val)

    script_path = job_dir / "inference.slurm"
    script_path.write_text(slurm_script)

    log.debug("uid=%s gid=%s user=%s", os.getuid(), os.getgid(), os.environ.get("USER", "(unset)"))
    log.debug("sbatch script: %s", script_path)

    env = os.environ.copy()
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True, env=env,
    )
    log.debug("sbatch rc=%s stdout=%r stderr=%r", result.returncode, result.stdout, result.stderr)

    if result.returncode != 0:
        return jsonify({"error": f"sbatch failed: {result.stderr}"}), 500

    slurm_job_id = result.stdout.strip().split()[-1]

    meta = {
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "sequence": summary,
    }
    (job_dir / "meta.json").write_text(json.dumps(meta))

    return jsonify({"job_id": job_id, "slurm_job_id": slurm_job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Job not found"}), 404

    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        return jsonify({"error": "Job metadata not found"}), 404
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Corrupt job metadata: {e}"}), 500

    output_dir = RESULTS_BASE / job_id
    cif_files = list(output_dir.rglob("*_model.cif"))
    json_files = list(output_dir.rglob("*_confidences*.json"))

    if cif_files:
        state = "completed"
    elif INFERENCE_MODE == "server":
        try:
            resp = http_requests.get(
                f"{INFERENCE_SERVER_URL}/status/{job_id}", timeout=5
            )
            state = resp.json().get("status", "queued") if resp.ok else "queued"
        except (http_requests.RequestException, ValueError):
            state = "queued"
    else:
        slurm_job_id = meta["slurm_job_id"]
        result = subprocess.run(
            ["squeue", "-j", slurm_job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        )
        slurm_state = result.stdout.strip()
        if slurm_state:
            state = slurm_state.lower()
        else:
            state = "failed"

    files = {
        "cif": [str(f.relative_to(output_dir)) for f in cif_files],
        "json": [str(f.relative_to(output_dir)) for f in json_files],
    }

    return jsonify({"job_id": job_id, "state": state, "files": files, "sequence": meta["sequence"]})


@app.route("/api/file/<job_id>/<path:filepath>")
def get_file(job_id, filepath):
    output_dir = RESULTS_BASE / job_id
    target = (output_dir / filepath).resolve()
    if not str(target).startswith(str(output_dir.resolve())):
        return jsonify({"error": "Invalid path"}), 403
    if not target.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(target)


@app.route("/api/jobs")
def list_jobs():
    jobs = []
    if not JOBS_DIR.exists():
        return jsonify(jobs)
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        meta_file = d / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            jobs.append({"job_id": meta["job_id"], "sequence": meta["sequence"][:40] + "..."})
    return jsonify(jobs)


@app.route("/api/debug")
def debug_info():
    if not DEBUG_ENABLED:
        return jsonify({"error": "Debug endpoint disabled"}), 403
    import pwd
    uid = os.getuid()
    gid = os.getgid()
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = f"(uid {uid} not in passwd)"

    sbatch_test = subprocess.run(
        ["sbatch", "--version"], capture_output=True, text=True,
    )
    squeue_test = subprocess.run(
        ["squeue", "--version"], capture_output=True, text=True,
    )

    return jsonify({
        "uid": uid,
        "gid": gid,
        "username": username,
        "HOME": os.environ.get("HOME", "(unset)"),
        "USER": os.environ.get("USER", "(unset)"),
        "template_vars": TEMPLATE_VARS,
        "sbatch_version": {"rc": sbatch_test.returncode, "out": sbatch_test.stdout.strip(), "err": sbatch_test.stderr.strip()},
        "squeue_version": {"rc": squeue_test.returncode, "out": squeue_test.stdout.strip(), "err": squeue_test.stderr.strip()},
    })


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=8060)
