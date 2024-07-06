#!/bin/bash -l
#SBATCH --job-name=FBLAS_fpga_compilation
#SBATCH --account p200301
#SBATCH --partition fpga
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --time 48:00:00
#SBATCH --output FBLAS_fpga_%x_%j.out
#SBATCH --error FBLAS_fpga_%x_%j.out

#Load software environment
module purge
module load env/staging/2023.1
module load 520nmx/20.4
module load ifpgasdk/20.4
module load GCC

cd /home/users/u101373/CG-FPGA/opencl
# Create folders
mkdir -p bin_fpga

#Compile
time aoc -board=p520_hpc_m210h_g3x16 -fp-relaxed -DINTEL_CL -report -fast-compile /home/users/u101373/CG-FPGA/opencl/src/FBGABlas_kernels/fblas_kernels_direct.cl -o bin/fpga/fblas_kernels_direct.fpga

