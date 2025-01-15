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

__global__ void localSort(int* arr, int n, int stage, int step) {

    extern __shared__ int sharedArr[];
    unsigned int globalId = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int localId = threadIdx.x;
    unsigned int offset = min(n>>1, blockDim.x);

    if (globalId < (n>>1)) {
        
        sharedArr[localId] = arr[globalId];
        sharedArr[localId + offset] = arr[globalId + (n>>1)];
        __syncthreads(); 

        do {
            while (step > 0) {

                unsigned int partner = globalId^step;
                unsigned int localPartner = partner - blockIdx.x*blockDim.x;
                if (partner > globalId) {
                    bool minmax = (globalId & stage) == 0;
                    swap(sharedArr, localId, localPartner, minmax ? sharedArr[localId] > sharedArr[localPartner] : sharedArr[localId] < sharedArr[localPartner]);
                } else {
                    globalId += n>>1;
                    partner += n>>1;
                    localId += offset;
                    localPartner += offset;

                    bool minmax = (globalId & stage) == 0;
                    swap(sharedArr, localId, localPartner, minmax ? sharedArr[localId] < sharedArr[localPartner] : sharedArr[localId] > sharedArr[localPartner]);

                    globalId -= n>>1;
                    localId -= offset;
                }
                step >>= 1;
                __syncthreads();
            }
            stage <<= 1;
            step = stage >> 1;
        } while (stage <= min(n, 1<<10));

        arr[globalId] = sharedArr[localId];
        arr[globalId + (n>>1)] = sharedArr[localId + offset];
        __syncthreads();
    }
}

void bitonicSort(Vector v){

    int n = v.n;
    int threads = n>>1;
    int blocks = (threads-1) / THREADS_PER_BLOCK+1;

    int* d_arr;
    int  size = n*sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, v.arr, size, cudaMemcpyHostToDevice);

    localSort<<<blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK*sizeof(int)>>>(d_arr, n, 1<<1, 1<<0);
    for(int stage=1<<11; stage<=n; stage<<=1) {
        for(int step=stage>>1; step>1<<9; step>>=1){
            
            bitonicExchange<<<blocks, THREADS_PER_BLOCK>>>(d_arr, threads, stage, step);
            cudaDeviceSynchronize();
        }
        localSort<<<blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK*sizeof(int)>>>(d_arr, n, stage, 1<<9);
    }

    cudaMemcpy(v.arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}