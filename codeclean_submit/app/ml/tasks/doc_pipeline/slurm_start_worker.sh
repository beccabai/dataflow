#!/bin/bash
#SBATCH --job-name=ray-worker
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH --requeue
#SBATCH --partition=vip_gpu_ailab_low
#SBATCH --account=ailab

set -xe

echo "Run Node: $SLURM_NODELIST"

# __doc_head_address_start__
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(dirname $(dirname $SLURM_SUBMIT_DIR))))
mkdir -p /dev/shm/suchenlin
rm -rf /dev/shm/suchenlin/PDF-Extract-Kit
cp -r /ailab/user/suchenlin/repo/PDF-Extract-Kit /dev/shm/suchenlin

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

srun --nodes=1 --ntasks=1 \
    ray start --address "$ip_head" \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block

sleep infinity