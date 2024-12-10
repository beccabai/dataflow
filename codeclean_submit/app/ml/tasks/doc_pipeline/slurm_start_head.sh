#!/bin/bash
#SBATCH --job-name=ray-header
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH --no-requeue
#SBATCH --partition=vip_gpu_ailab
#SBATCH --account=ai4phys

set -xe

# __doc_head_address_start__
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(dirname $(dirname $SLURM_SUBMIT_DIR))))
mkdir -p /dev/shm/suchenlin
rm -rf /dev/shm/suchenlin/PDF-Extract-Kit
cp -r /ailab/user/suchenlin/repo/PDF-Extract-Kit /dev/shm/suchenlin

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --resources '{"headNode": 100000}' --block --dashboard-host=0.0.0.0 --disable-usage-stats
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep infinity