export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

srun \
  --partition=MoE \
  --mpi=pmi2 \
  --job-name=internvl_3 \
  -c 128 \
  --gres=gpu:8 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --quotatype=spot \
  python internvl3_qa.py