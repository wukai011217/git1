#!/bin/bash
#SBATCH --job-name=calc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 加载必要的模块
module load vasp

# 运行VASP
mpirun -np $SLURM_NTASKS vasp_std
