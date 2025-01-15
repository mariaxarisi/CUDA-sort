#include "../include/vector.h"
#include "../include/bitonic.h"

#include <cuda_runtime.h>
#include <stdbool.h>

#define THREADS_PER_BLOCK 1024

__device__ void swap(int* arr, int i, int j, bool condition) {
    
    if(condition){
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

__global__ void bitonicExchange(int* arr, int threads, int stage, int step) {

    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid < threads) {

        unsigned int partner = tid^step;
        if (partner > tid) {
            bool minmax = (tid & stage) == 0;
            swap(arr, tid, partner, minmax ? arr[tid] > arr[partner] : arr[tid] < arr[partner]);
        } else {
            tid += threads;
            partner += threads;

            bool minmax = (tid & stage) == 0;
            swap(arr, tid, partner, minmax ? arr[tid] < arr[partner] : arr[tid] > arr[partner]);
        }
    }
}

void bitonicSort(Vector v) {

    int n = v.n;
    int threads = n>>1;
    int blocks = (threads-1) / THREADS_PER_BLOCK+1;

    int* d_arr;
    int  size = n*sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, v.arr, size, cudaMemcpyHostToDevice);

    for (int stage=2; stage<=n; stage<<=1) {
        for (int step=stage>>1; step>0; step>>=1) {
            bitonicExchange<<<blocks, THREADS_PER_BLOCK>>>(d_arr, threads, stage, step);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(v.arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}