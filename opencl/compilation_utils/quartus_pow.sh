#!/bin/bash -l
#SBATCH --job-name=QuasrtusPowLog
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
#SBATCH --mail-user=daniele6.ferrario@mail.polimi.it
#SBATCH --mail-type=ALL

#Load software environment
module purge
module load intel-fpga 

cd /home/users/u101373/CG-FPGA/opencl/bin/global_memory_kernels

#Compile
time quartus_pow top

