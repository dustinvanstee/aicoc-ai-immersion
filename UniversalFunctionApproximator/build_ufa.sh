#!/bin/bash

module load ibm-wml-ce/1.7.0-1
module load gcc/7.4.0
module load cmake/3.18.2
module load spectrum-mpi/10.3.1.2-20200121
conda activate wmlce17-ornl

if [ -d "build" ]
then 
  echo "Removing old build files"
  rm -rf build
fi

echo "Create new build directory"
mkdir -p build
cd build
cmake  -DCMAKE_CXX_COMPILER=mpiCC -DCMAKE_PREFIX_PATH=/ccs/home/vanstee/.conda/envs/powerai-ornl/lib/python3.6/site-packages/torch ..
cmake --build . --config Release
# cmake  -DCMAKE_PREFIX_PATH=/ccs/home/vanstee/.conda/envs/powerai-ornl/lib/python3.6/site-packages/torch ..
cp ufa ../ufa
echo "Built ufa to use run :"
echo   ufa pytorch_model


