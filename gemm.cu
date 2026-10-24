#include <cuda_runtime.h>
#include <stddef.h>

// 每个block计算一个16x16的输出tile，通过共享内存复用A和B的数据
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K){
    const int tileSize = 16;
    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    float sum = 0.0f;
    int tileCount = (K - 1) / tileSize + 1;
    for(int tile = 0; tile < tileCount; tile++){
        int aCol = tile * tileSize + threadIdx.x;
        int bRow = tile * tileSize + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < tileSize; k++){
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N){
        C[(size_t)row * N + col] = sum;
    }
}

// A[M,K] * B[K,N] = C[M,N]，矩阵按行优先存储
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K){
    if(M <= 0 || N <= 0 || K <= 0){
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N - 1) / 16 + 1, (M - 1) / 16 + 1);
    gemm_tiled<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
