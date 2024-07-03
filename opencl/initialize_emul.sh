#!/bin/bash

# Update submodules
echo " "
echo "Update submodules..."
git submodule update --init
cd FBLAS
git submodule update --init
module load Python
pip install -r codegen/requirements.txt
cd ..

# Load modules
echo " "
echo "Load needed modules..."
module purge
module load env/staging/2023.1
module load ifpgasdk/20.4
module load 520nmx/20.4

# Compile the emulation kernels
echo " "
echo "Compile emulation kernels..."
cd device
chmod +x compile_emul.sh
./compile_emul.sh
cd ..

# Add an io folder
mkdir -p io

# Finally we need to compile the host code
echo " "
echo "Compile host program..."
make

echo " "
echo "Initialization done!"