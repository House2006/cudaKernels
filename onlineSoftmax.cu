#include <cuda_runtime.h>
#include <math.h>

__device__ void onlineSoftmax_update(float value, float* maxValue, float* sumValue){
    if(*sumValue == 0.0f){
        *maxValue = value;
        *sumValue = 1.0f;
        return;
    }

    float newMax = fmaxf(*maxValue, value);
    *sumValue = *sumValue * __expf(*maxValue - newMax) + __expf(value - newMax);
    *maxValue = newMax;
}

__device__ void onlineSoftmax_merge(float maxA, float sumA, float maxB, float sumB,
                                    float* maxValue, float* sumValue){
    if(sumA == 0.0f){
        *maxValue = maxB;
        *sumValue = sumB;
        return;
    }
    if(sumB == 0.0f){
        *maxValue = maxA;
        *sumValue = sumA;
        return;
    }

    float newMax = fmaxf(maxA, maxB);
    *sumValue = sumA * __expf(maxA - newMax) + sumB * __expf(maxB - newMax);
    *maxValue = newMax;
}

// 每个block在线计算一组数据对应的(max, sum)
__global__ void onlineSoftmax_partial(const float* input, int N,
                                      float* partialMax, float* partialSum){
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float maxValue = -INFINITY;
    float sumValue = 0.0f;
    for(int idx = index; idx < N; idx += stride){
        onlineSoftmax_update(input[idx], &maxValue, &sumValue);
    }

    __shared__ float smemMax[256];
    __shared__ float smemSum[256];
    smemMax[tid] = maxValue;
    smemSum[tid] = sumValue;
    __syncthreads();

    for(int step = blockDim.x / 2; step > 0; step /= 2){
        if(tid < step){
            float mergedMax, mergedSum;
            onlineSoftmax_merge(smemMax[tid], smemSum[tid],
                                smemMax[tid + step], smemSum[tid + step],
                                &mergedMax, &mergedSum);
            smemMax[tid] = mergedMax;
            smemSum[tid] = mergedSum;
        }
        __syncthreads();
    }

    if(tid == 0){
        partialMax[blockIdx.x] = smemMax[0];
        partialSum[blockIdx.x] = smemSum[0];
    }
}

// 合并各block的在线统计量
__global__ void onlineSoftmax_reduce(const float* partialMax, const float* partialSum,
                                     int partialCount, float* maxValue, float* sumValue){
    int tid = threadIdx.x;
    float localMax = -INFINITY;
    float localSum = 0.0f;
    for(int idx = tid; idx < partialCount; idx += blockDim.x){
        float mergedMax, mergedSum;
        onlineSoftmax_merge(localMax, localSum, partialMax[idx], partialSum[idx],
                            &mergedMax, &mergedSum);
        localMax = mergedMax;
        localSum = mergedSum;
    }

    __shared__ float smemMax[256];
    __shared__ float smemSum[256];
    smemMax[tid] = localMax;
    smemSum[tid] = localSum;
    __syncthreads();

    for(int step = blockDim.x / 2; step > 0; step /= 2){
        if(tid < step){
            float mergedMax, mergedSum;
            onlineSoftmax_merge(smemMax[tid], smemSum[tid],
                                smemMax[tid + step], smemSum[tid + step],
                                &mergedMax, &mergedSum);
            smemMax[tid] = mergedMax;
            smemSum[tid] = mergedSum;
        }
        __syncthreads();
    }

    if(tid == 0){
        *maxValue = smemMax[0];
        *sumValue = smemSum[0];
    }
}

__global__ void onlineSoftmax_norm(const float* input, float* output, int N,
                                   const float* maxValue, const float* sumValue){
    float max = __ldg(maxValue);
    float sum = __ldg(sumValue);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int idx = index; idx < N; idx += stride){
        output[idx] = __expf(input[idx] - max) / sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N){
    if(N <= 0){
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N - 1) / threadsPerBlock + 1;
    if(blocksPerGrid > 4 * 108){
        blocksPerGrid = 4 * 108;
    }

    float *partialMax, *partialSum, *maxValue, *sumValue;
    cudaMalloc(&partialMax, sizeof(float) * blocksPerGrid);
    cudaMalloc(&partialSum, sizeof(float) * blocksPerGrid);
    cudaMalloc(&maxValue, sizeof(float));
    cudaMalloc(&sumValue, sizeof(float));

    onlineSoftmax_partial<<<blocksPerGrid, threadsPerBlock>>>(input, N, partialMax, partialSum);
    onlineSoftmax_reduce<<<1, threadsPerBlock>>>(partialMax, partialSum, blocksPerGrid,
                                                maxValue, sumValue);
    onlineSoftmax_norm<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, maxValue, sumValue);
    cudaDeviceSynchronize();

    cudaFree(partialMax);
    cudaFree(partialSum);
    cudaFree(maxValue);
    cudaFree(sumValue);
}
