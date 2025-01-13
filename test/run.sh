#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

nvidia-smi
module load gcc/13.2.0-iqpfkya cuda/12.4.0-zk32gam

cd /home/c/charisim/cuda-sort
rm ./test/results.txt
touch ./test/results.txt

make clean
make

for n in {16..31};
do
    echo -e "N = $n\n" >> ./test/results.txt
    make run $n >> ./test/results.txt
    echo -e "\n" >> ./test/results.txt
done