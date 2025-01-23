#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

nvidia-smi
module load gcc/12.2.0
module load cuda/12.2.1-bxtxsod

cd /home/c/charisim/cuda-sort
rm ./test/TESLAP100.txt
touch ./test/TESLAP100.txt

make clean
make

for n in {16..28};
do
    echo -e "N = $n\n" >> ./test/TESLAP100.txt
    make run $n >> ./test/TESLAP100.txt
    echo -e "\n" >> ./test/TESLAP100.txt
done