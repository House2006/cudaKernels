#include <stdio.h>

__global__ void myKernel() 
{
    printf("Hello, world from the device!\n"); 
} 

int main() 
{ 
    myKernel<<<4,4>>>(); 
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
        }
        else {
            printf("No CUDA error\n");        }
     cudaDeviceSynchronize();
} 
