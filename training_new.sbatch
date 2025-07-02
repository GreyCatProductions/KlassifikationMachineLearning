#!/bin/bash
#SBATCH --job-name=setfit_training
#SBATCH --partition=gpu              # On the 'gpu' partition
#SBATCH --nodes=1                    # On a single node
#SBATCH --ntasks=1                   # With a single task (this should always be 1 for single-GPU jobs)
#SBATCH --cpus-per-task=4            # With that many CPU cores (adjust if your model benefits from more CPU during data prep)
#SBATCH --time=5-00:00               # Maximum runtime of the job as "d-hh:mm" (1 day, 0 hours, 0 minutes)
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem-per-gpu=40GB


# --- Environment Setup ---
echo "Activating Python environment..."
source ./my_env/bin/activate
echo "Environment activated."

# --- GPU and Node Diagnostics ---
echo "--- GPU Status ---"
nvidia-smi
echo "--- Running on Host ---"
echo "Running on $(hostname)"
echo "--- Current Directory ---"
echo "Current working directory: $(pwd)"
echo "--- Temporary Directories Debug ---"
echo "SLURM_TMPDIR is: ${SLURM_TMPDIR}" # Will likely still be empty, but good to confirm
echo "TMPDIR is: ${TMPDIR}"             # This will likely point to /tmp
echo "Checking /tmp content:"
ls -ld /tmp                            # List permissions and check if it's the right path
echo "Checking /dev/shm content:"
ls -ld /dev/shm                        # List permissions and check if it's the right path
echo "---"

# --- Model Staging for Fast Loading ---
# Model path
MODEL_SHARED_NETWORK_PATH="./downloaded_model"

# Working temp dir
MODEL_LOCAL_SCRATCH_BASE="/tmp"

# Create a unique directory within /tmp for your job to avoid conflicts
MODEL_LOCAL_SCRATCH_DEST="${MODEL_LOCAL_SCRATCH_BASE}/${USER}_setfit_model_cache_${SLURM_JOB_ID}"


echo "Staging model from ${MODEL_SHARED_NETWORK_PATH} to local scratch ${MODEL_LOCAL_SCRATCH_DEST}..."
mkdir -p "${MODEL_LOCAL_SCRATCH_DEST}" # Ensure the destination directory exists
# Ensure proper error handling for mkdir, just in case
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create local scratch directory ${MODEL_LOCAL_SCRATCH_DEST}. Aborting."
    exit 1
fi
# Use rsync for robust copying. '-a' preserves permissions, '-v' for verbose output.
# The trailing slash on MODEL_SHARED_NETWORK_PATH/ is important to copy contents, not the directory itself.
rsync -av "${MODEL_SHARED_NETWORK_PATH}/" "${MODEL_LOCAL_SCRATCH_DEST}/"
echo "Model staging complete."

# --- Set Environment Variable for Your Python Script ---
# Your Python code (specifically the `load_model` function) will read this variable
export SETFIT_MODEL_PATH=${MODEL_LOCAL_SCRATCH_DEST}
echo "SETFIT_MODEL_PATH set to: ${SETFIT_MODEL_PATH}"

# --- Run Your Python Script ---
echo "Starting Few_Shot_Training.py..."
python3 -u Few_Shot_Training.py
echo "Few_Shot_Training.py finished."