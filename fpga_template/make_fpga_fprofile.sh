#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --account=p200301
#SBATCH --partition=fpga
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -o make_fpga_fprofile.out 
#SBATCH --mail-user=daniele6.ferrario@mail.polimi.it

module load env/staging/2023.1
module load intel
module load 520nmx/20.4
module load intel-fpga

echo $PATH
cd /home/users/u101373/CG-FPGA/fpga_template/build_fprofile
time make fpga 
