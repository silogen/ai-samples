#!/bin/bash
#SBATCH --job-name=pfui-gpu
#SBATCH --partition=CHANGE_ME          # EDIT: your SLURM partition with GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:mi300:1             # EDIT: your cluster's GPU gres name/count
#SBATCH --time=08:00:00

set -euo pipefail

# Resolve the repo root to the directory you ran `sbatch` from.
# (${BASH_SOURCE[0]} points at SLURM's spool dir inside a job, so we can't use it.)
PFUI_DIR="${SLURM_SUBMIT_DIR}"
source "${PFUI_DIR}/config.env"

INFERENCE_PORT="${INFERENCE_PORT:-8061}"
INF_CONTAINER="pfui-inference-${SLURM_JOB_ID}"
UI_CONTAINER="pfui-server-${SLURM_JOB_ID}"

mkdir -p "${PFUI_DIR}/jobs"

cleanup() {
  echo "Stopping containers..."
  docker stop "${UI_CONTAINER}" 2>/dev/null || true
  docker stop "${INF_CONTAINER}" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "=== PFUI GPU Server ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Starting:  $(date)"
echo ""
echo "============================================"
echo "  To access the UI, set up an SSH tunnel:"
echo ""
echo "  ssh -N -L ${PFUI_PORT}:${SLURMD_NODENAME}:${PFUI_PORT} $(whoami)@$(hostname -f)"
echo ""
echo "  Then open: http://localhost:${PFUI_PORT}"
echo "============================================"
echo ""

JOBS_LOGS="${PFUI_DIR}/jobs/slurm_logs"
RESULTS_JOBS="${RESULTS_DIR}/pfui_jobs"
mkdir -p "${RESULTS_JOBS}" "${JOBS_LOGS}"
docker run --rm \
  -v "${RESULTS_JOBS}:/fix/results" \
  -v "${JOBS_LOGS}:/fix/logs" \
  alpine chown -R "$(id -u):$(id -g)" /fix

echo "=== Building OpenFold3 Docker image ==="
docker build -t "${DOCKER_IMAGE}" -f "${OF3_DOCKERFILE}" "$(dirname "${OF3_DOCKERFILE}")"

echo "=== Starting Inference Server ==="
docker run --rm \
  --name "${INF_CONTAINER}" \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE \
  --shm-size 64G \
  -v "${PFUI_DIR}/inference_server.py:/app/inference_server.py:ro" \
  -v "${PFUI_DIR}/configs/OpenFold3/inference_config_pfui.yml:/config/inference_config_pfui.yml:ro" \
  -v "${RESULTS_DIR}:/results" \
  -v "${PFUI_DIR}/jobs:/jobs" \
  -v "${CACHE_DIR}:/cache" \
  -e OPENFOLD_CACHE="/cache/openfold3" \
  -e HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}" \
  -e RUNNER_YAML="/config/inference_config_pfui.yml" \
  -e OUTPUT_DIR="/results/pfui_jobs" \
  -e INFERENCE_PORT="${INFERENCE_PORT}" \
  -e NUM_DIFFUSION_SAMPLES=1 \
  ${DOCKER_IMAGE} \
  /opt/venv/bin/python /app/inference_server.py &

INF_PID=$!

echo "Waiting for inference server to load model..."
for i in $(seq 1 120); do
  if ! kill -0 $INF_PID 2>/dev/null; then
    echo "ERROR: Inference server process died."
    exit 1
  fi
  HEALTH=$(curl -s "http://localhost:${INFERENCE_PORT}/health" 2>/dev/null) || true
  if echo "$HEALTH" | grep -q '"model_ready".*true'; then
    echo "Inference server ready!"
    break
  fi
  sleep 5
done

if ! echo "$HEALTH" | grep -q '"model_ready".*true'; then
  echo "ERROR: Inference server did not become ready in 10 minutes."
  exit 1
fi

echo "=== Building UI Docker image ==="
docker build -t pfui:latest -f "${PFUI_DIR}/Dockerfile.server" "${PFUI_DIR}"

echo "=== Starting UI container ==="
docker run --rm \
  --name "${UI_CONTAINER}" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e INFERENCE_MODE=server \
  -e INFERENCE_SERVER_URL="http://localhost:${INFERENCE_PORT}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "${PFUI_DIR}/jobs:/jobs" \
  -v "${RESULTS_DIR}:/results" \
  pfui:latest &

wait -n
wait
