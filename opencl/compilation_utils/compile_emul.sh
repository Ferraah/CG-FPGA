echo " "
echo "Reloading needed modules..."

module purge
module load env/staging/2023.1
module load ifpgasdk/20.4
module load 520nmx/20.4

echo " "
echo "Compiling for emulation..."
echo "  board=p520_max_sg280l"
echo "  output=bin_emul/fblas_routines.aocx"

aoc -march=emulator -board=p520_max_m210h_g3x16 -legacy-emulator daxpy.cl ddot.cl dscal.cl dgemv.cl

mv dgemv bin_emul/fblas_routines
mv dgemv.aocx bin_emul/fblas_routines.aocx
mv bin_emul/fblas_routines/dgemv.bc bin_emul/fblas_routines/fblas_routines.bc
mv bin_emul/fblas_routines/dgemv.log bin_emul/fblas_routines/fblas_routines.log

echo "End compilation of the routines: daxpy, ddot, dscal, dgemv"