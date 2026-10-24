#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)           \
    do{                            \
        cudaError_t error = call;  \
        if(error!=cudaSuccess){    \
            fprintf(stderr,"CUDA error: %s:%d, %s\n",__FILE__,__LINE__,cudaGetErrorString(error));    \
            exit(EXIT_FAILURE);    \
        }                          \
    }while(0)

#define CUDA_CHECK_LAUNCH() \
    do{ \
        cudaError_t error = cudaGetLastError(); \
        if(error!=cudaSuccess){ \
            fprintf(stderr,"CUDA kernel launch error: %s:%d, %s\n",__FILE__,__LINE__,cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    }while(0);

#define CUDA_CHECK_EXEC() \
    do{ \
        cudaError_t error = cudaDeviceSynchronize(); \
        if(error!=cudaSuccess){ \
            fprintf(stderr,"CUDA kernel execution error: %s:%d, %s\n",__FILE__,__LINE__,cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    }while(0)
