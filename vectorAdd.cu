#include <cstdio>
#include <cstdlib>
#include "cuda_header.cuh"


__global__ void  vectorAdd(float* A,float* B,float*C,int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index<N){
        C[index] = A[index] + B[index];
    }
}

void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc,char** argv){
   
    cudaEvent_t  start, stop,kernel_start,kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    int N = atoi(argv[1]);
    int size = N*sizeof(float);
    
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    for(int i=0;i<N;i++){
        h_A[i] = (float)i;
        h_B[i] = (float)(i*2);
    }
    
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaEventRecord(start);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (N+blockSize-1)/blockSize;

    cudaEventRecord(kernel_start);
    vectorAdd<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);
    cudaEventRecord(kernel_stop);   

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    for(int i=0;i<N;i++){
        if(fabsf(h_C[i]-(h_A[i]+h_B[i]))>1e-5f){
            printf("Verification FAILED at index %d!\n", i);
            return -1;
        }
    }
    printf("Verification PASSED!\n");

    float* h_C_cpu = (float*)malloc(size);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU time: %.4f ms\n", cpu_ms);
    
    float gpu_total_ms = 0, kernel_ms = 0;
    cudaEventElapsedTime(&gpu_total_ms, start, stop);
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
    printf("GPU kernel time: %.4f ms\n", kernel_ms);
    printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Speedup (end-to-end): %.1fx\n", cpu_ms / gpu_total_ms);

    // ========== 计算有效带宽 ==========
    // 向量加法读 2 个 float、写 1 个 float = 12 bytes/element
    float bandwidth_gb = (3.0f * size) / (kernel_ms * 1e-3f) / 1e9f;
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gb);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;


}



__global__ void vector_add(
    const float* A, const float* B, float* C, int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
     
    float4 *A4 = (float4 *)A;
    float4 *B4 = (float4 *)B;
    float4 *C4 = (float4 *)C;
    float4 a4,b4;
    
    int n4 = N/4;
    for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
        a4 = A4[idx];
        b4 = B4[idx];
        a4.x += b4.x;
        a4.y += b4.y;
        a4.z += b4.z;
        a4.w += b4.w;
        C4[idx] = a4;         
    }
    if(index < N - n4*4){
        int idx = n4*4 + index;
        C[idx] = A[idx] + B[idx]; 
    }
}

 