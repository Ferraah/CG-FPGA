#!/bin/bash -l
#SBATCH --job-name=FBLAS_fpga_compilation
#SBATCH --account p200301
#SBATCH --partition fpga
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --time 01:00:00
#SBATCH --output run_5k_1f.out
#SBATCH --error run_5k_1f.out

#Load software environment
module_load_fpga

cd /home/users/u101373/CG-FPGA/opencl/
# Create folders

make -j && srun -n 1 ./cg_opencl
