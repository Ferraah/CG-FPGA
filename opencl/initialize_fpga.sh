#!/bin/bash

# Load modules
echo " "
echo "Load needed modules..."
module purge
module load env/staging/2023.1
module load ifpgasdk/20.4
module load 520nmx/20.4

# Finally we need to compile the host code
echo " "
echo "Compile host program..."
make

echo " "
echo "Initialization done!"