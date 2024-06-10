#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --account=p200301
#SBATCH --partition=fpga
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module load env/staging/2023.1
module load intel-fpga/2023.1.0
module load intel
module load CMake
module load 520nmx
cd /home/users/u101373/dot_product/build
make 
make report