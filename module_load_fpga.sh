#!/bin/sh

function module_load_fpga(){
	module load env/staging/2023.1
	module load CMake 
	module load intel-fpga
	module load 520nmx/20.4
	module load OpenMPI
}
