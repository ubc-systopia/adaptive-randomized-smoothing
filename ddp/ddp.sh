#!/bin/bash
#SBATCH --account=<>
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=output-cc/ddp.out


module load python/3.10 cuda cudnn scipy-stack
source /home/<>/venv-ars/bin/activate
export MASTER_ADDR=$(hostname)

tar xf /scratch/<>/cache_imagenet.tar -C $SLURM_TMPDIR
tar xf /scratch/<>/cache_imagenet_val.tar -C $SLURM_TMPDIR

srun python ddp/train_ddp.py -s 1 -y src/configs/imagenet_ars.yaml --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) 