#!/bin/bash
#SBATCH --job-name=ray-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpugpu
#SBATCH --no-requeue
#SBATCH --partition=vip_gpu_ailab
#SBATCH --account=ai4phys

set -xe

# __doc_head_address_start__
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(dirname $(dirname $SLURM_SUBMIT_DIR))))

port=8265
RAY_ADDRESS=http://$head_node_ip:$port
export RAY_ADDRESS
echo "Ray Address: $RAY_ADDRESS"

srun ray job submit --no-wait -- $@