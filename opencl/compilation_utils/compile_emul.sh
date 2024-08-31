module purge
module load env/staging/2023.1
module load ifpgasdk/20.4
module load 520nmx/20.4

cd /home/users/u101373/CG-FPGA/opencl/src/global_memory_kernels/

time aoc -march=emulator -board=p520_max_m210h_g3x16 -legacy-emulator direct_daxpy.cl direct_ddot.cl direct_gemv.cl
