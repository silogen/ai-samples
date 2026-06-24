#!/bin/bash
#SBATCH --job-name=pfui-inference-{{JOB_ID}}
#SBATCH --partition=CHANGE_ME          # EDIT: your SLURM partition with GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:mi300:1             # EDIT: your cluster's GPU gres name/count
#SBATCH --time=08:00:00
#SBATCH --output={{PFUI_DIR}}/jobs/slurm_logs/pfui-inference-{{JOB_ID}}_%j.out
#SBATCH --error={{PFUI_DIR}}/jobs/slurm_logs/pfui-inference-{{JOB_ID}}_%j.err

CONTAINER_NAME="pfui-inference-{{JOB_ID}}-${SLURM_JOB_ID}"

cleanup() {
  echo "Stopping inference container..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null
}
trap cleanup EXIT SIGTERM SIGINT

echo "=== PFUI Inference Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"

OUTPUT_DIR="{{RESULTS_DIR}}/pfui_jobs/{{JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting OpenFold3 Inference ==="
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE \
  --shm-size 64G \
  -v {{PFUI_DIR}}/jobs/{{JOB_ID}}:/query \
  -v {{PFUI_DIR}}/configs/OpenFold3/inference_config_pfui.yml:/config/inference_config_pfui.yml:ro \
  -v {{RESULTS_DIR}}:/results \
  -v {{CACHE_DIR}}:/cache \
  -e OPENFOLD_CACHE="/cache/openfold3" \
  -e HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}" \
  {{DOCKER_IMAGE}} \
  bash -c 'yes | /opt/venv/bin/run_openfold predict \
    --query_json /query/query.json \
    --output_dir /results/pfui_jobs/{{JOB_ID}}/ \
    --use_msa_server={{USE_MSA_SERVER}} \
    --num_diffusion_samples=1 \
    --runner_yaml=/config/inference_config_pfui.yml' \
  2>&1 &

wait $!

echo "=== OpenFold3 Inference Completed ==="
echo "Finished at: $(date)"
