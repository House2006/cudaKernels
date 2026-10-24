#include <cuda_runtime.h>
__device__ __forceinline__ float silu(float x){
    if(x>=0.0f){
        return __fdividef(x, 1.0f+__expf(-x));
    }else{
        float e = __expf(x);
        return __fdividef(x*e, 1.0f+e);
    }
}

 
__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
     
    if(halfN%4==0){
        float4 *input4 = (float4 *)input;     
        float4 *output4 = (float4 *)output;
        float4 x4,y4;
        
        int n4 = halfN/4;
        for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
            x4 = input4[idx];
            y4 = input4[n4+idx];
            x4.x = silu(x4.x) * y4.x;
            x4.y = silu(x4.y) * y4.y;
            x4.z = silu(x4.z) * y4.z;
            x4.w = silu(x4.w) * y4.w;
            output4[idx] = x4;         
        }
    }else{
        for(int idx=index;idx<halfN;idx+=gridDim.x*blockDim.x){
            output[idx] = silu(input[idx])*input[idx+halfN];
        }
    }

    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
