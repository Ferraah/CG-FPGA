#!/bin/bash -l
#SBATCH --job-name=lucagu_ouptut
#SBATCH --account p200301
#SBATCH --partition fpga
#SBATCH --qos default
#SBATCH --nodes 4
#SBATCH --time 03:00:00
#SBATCH --output lucagu.out
#SBATCH --error lucagu.out

#Load software environment
module_load_fpga

cd /home/users/u101373/CG-FPGA/opencl/
# Create folders

make -j 

# 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1088, 1152, 1216, 1280
# Number of iterations for benchmarking
num_iter=5

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests"

p_1=(1000 5000)
p_2=(1000 5000 10000)
p_4=(1000 5000 10000)
p_8=(1000 5000 10000 20000)
#p_16=(1000 5000 10000 20000 30000)

echo "num_proc: 1"
for matrix_size in "${p_1[@]}"; do
    srun -n 1 ./cg_opencl $matrix_size $num_iter
done
sleep 10 
sacct -o JobID,JobName,Partition,State,AllocCPUs,ConsumedEnergy,Elapsed,TotalCPU,NodeList | tail -2

echo "num_proc: 2"
for matrix_size in "${p_2[@]}"; do
    srun -n 2 ./cg_opencl $matrix_size $num_iter
done
sleep 10 
sacct -o JobID,JobName,Partition,State,AllocCPUs,ConsumedEnergy,Elapsed,TotalCPU,NodeList | tail -3

echo "num_proc: 4"
for matrix_size in "${p_4[@]}"; do
    srun -n 4 ./cg_opencl $matrix_size $num_iter
done
sleep 10 
sacct -o JobID,JobName,Partition,State,AllocCPUs,ConsumedEnergy,Elapsed,TotalCPU,NodeList | tail -3

#echo "num_proc: 8"
#for matrix_size in "${p_8[@]}" ; do
#    srun -n 8 ./cg_opencl $matrix_size $num_iter
#done
#sleep 10 
#sacct -o JobID,JobName,Partition,State,AllocCPUs,ConsumedEnergy,Elapsed,TotalCPU,NodeList | tail -4




echo "Finished!"