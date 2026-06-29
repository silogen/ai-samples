#!/bin/bash
#SBATCH --job-name=pfui-server
#SBATCH --partition=CHANGE_ME          # EDIT: your SLURM partition (CPU-only is fine)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

set -euo pipefail

# Resolve the repo root to the directory you ran `sbatch` from.
# (${BASH_SOURCE[0]} points at SLURM's spool dir inside a job, so we can't use it.)
PFUI_DIR="${SLURM_SUBMIT_DIR}"
source "${PFUI_DIR}/config.env"

CONTAINER_NAME="pfui-server-${SLURM_JOB_ID}"

mkdir -p "${PFUI_DIR}/jobs"

cleanup() {
  echo "Stopping UI container..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "=== PFUI Server ==="
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

echo "=== Building UI Docker image ==="
docker build -t pfui:latest -f "${PFUI_DIR}/Dockerfile.server" "${PFUI_DIR}"

echo "=== Starting UI container ==="
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e PFUI_DIR="${PFUI_DIR}" \
  -e RESULTS_DIR="${RESULTS_DIR}" \
  -e CACHE_DIR="${CACHE_DIR}" \
  -e DOCKER_IMAGE="${DOCKER_IMAGE}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "${PFUI_DIR}/jobs:/jobs" \
  -v "${RESULTS_DIR}:/results" \
  -v /usr/bin/sbatch:/usr/bin/sbatch:ro \
  -v /usr/bin/squeue:/usr/bin/squeue:ro \
  -v /usr/bin/scancel:/usr/bin/scancel:ro \
  -v /usr/lib/x86_64-linux-gnu/slurm:/usr/lib/x86_64-linux-gnu/slurm:ro \
  -v /usr/lib/x86_64-linux-gnu/libmunge.so.2.0.0:/usr/lib/x86_64-linux-gnu/libmunge.so.2.0.0:ro \
  -v /usr/lib/x86_64-linux-gnu/libmunge.so.2:/usr/lib/x86_64-linux-gnu/libmunge.so.2:ro \
  -v /usr/lib/x86_64-linux-gnu/libmunge.so:/usr/lib/x86_64-linux-gnu/libmunge.so:ro \
  -v /etc/slurm:/etc/slurm:ro \
  -v /var/run/munge:/var/run/munge \
  pfui:latest &

wait $!
