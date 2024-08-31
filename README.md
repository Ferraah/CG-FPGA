# Conjugate Gradient implementation on Meluxina FPGAs

The following project collects the code employed in the benchmarking of the Conjugate Method on Meluxina FPGAs,
in the context of the "Advanced Computer Architectures" course held at PoliMi during the a.y. 2023/2024.


![Intel Stratix 10](https://www.intel.com/content/dam/www/central-libraries/us/en/images/stratix10mxboardtop.jpg.rendition.intel.web.864.486.jpg "Intel Stratix 10")

## Compilation
The code was intended to be specifically tailored to the author needs, but still it can be compiled to be used from other Meluxina users.
The framework employed is Intel HLS tools along with OpenCL.

### Device code/Kernels compilation
To compile the FPGA kernels into binaries, follow the instructions in `compilation_utils/compile_fpga.sh`. This step could take from 5 to 10 hours to complete.

### Host code compilation
In `src/main.cpp`, specify the directory containing the matrices and rhs following the requested file name format. In `src/host_code.cpp`, define in both `prepare` methods the path to the FPGA binaries (.aocx files).

## Run 
```
cd opencl
mkdir build
make -j
./cg_opencl {size_of_matrix} {repetitions of runs}
```
