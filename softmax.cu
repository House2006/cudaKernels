 #include <cuda_runtime.h>
__device__ float warpReduce(float val){
    for(int offset = 16;offset > 0;offset /= 2){
        val += __shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__device__ float warpMax(float max){
    for(int offset = 16;offset > 0;offset /= 2){
        max = fmaxf(__shfl_down_sync(0xffffffff,max,offset),max);
    }
    return max;
}

__global__ void softmax_partialmax(const float* input, int N, float* partial){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid&31; //tid%32;
    int wid = tid>>5;//tid/32;

    float4* input4 =  (float4 *)input;
    int n4 = N>>2;

    float max = -INFINITY;
    float4 data;
    float dat;
    for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
        data = input4[idx];
        max = fmaxf(data.x,max);
        max = fmaxf(data.y,max);
        max = fmaxf(data.z,max);
        max = fmaxf(data.w,max);
    }
    int tail = n4<<2;
    for(int idx=tail+index;idx<N;idx+=gridDim.x*blockDim.x){
        max = fmaxf(input[idx],max);
    }

    __shared__ float smem[32];
    max = warpMax(max);
    if(lane == 0){
        smem[wid] = max;
    }
    __syncthreads();

    int warpNums = blockDim.x>>5;
    if(wid == 0){
        max = lane<warpNums?smem[lane]:-INFINITY;
        max = warpMax(max);  //blockDim <= 1024
    }

    if(tid==0) {
        partial[blockIdx.x] = max;
    }
}

__global__ void softmax_max(float *partial, int n, float *maxmum){
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    float max = -INFINITY;
    __shared__ float smem[32];
    for(int idx = tid;idx < n;idx += blockDim.x){
        max = fmaxf(partial[idx],max);
    }
    max = warpMax(max);
    if(lane==0){
        smem[wid] = max;
    }
    __syncthreads();
    int warpNums = blockDim.x >>5;
    if(wid==0){
        max = lane<warpNums?smem[lane]:-INFINITY;
        max = warpMax(max);
    }
    if(tid==0){
        *maxmum = max;        
    }
}
__global__ void softmax_reduce(const float* input, float* output, int N, float* sum, float* max) {

    float maxmum = __ldg(max);
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid%32;
    int wid = tid/32;

    float4* input4 =  (float4 *)input;
    float4* output4 =  (float4 *)output;
    int n4 = N/4;

    float val = 0.0f;
    float4 data;
    float dat;
    for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
        data = input4[idx];
        data.x = __expf(data.x-maxmum);
        data.y = __expf(data.y-maxmum);
        data.z = __expf(data.z-maxmum);
        data.w = __expf(data.w-maxmum);
        output4[idx] = data;
        val += data.x + data.y + data.z + data.w;

    }
    int tail = n4*4;
    for(int idx=tail+index;idx<N;idx+=gridDim.x*blockDim.x){
        dat = __expf(input[idx]-maxmum);
        output[idx] = dat;
        val += dat;
    }

    __shared__ float smem[32];
    val = warpReduce(val);
    if(lane == 0){
        smem[wid] = val;
    }
    __syncthreads();

    int warpNums = blockDim.x/32;
    if(wid == 0){
        val = lane<warpNums?smem[lane]:0;
        val = warpReduce(val);  //blockDim <= 1024
    }

    if(tid==0) {
        atomicAdd(sum, val);
    }
}

__global__ void softmax_norm(float *input,int N,float *sum){
    float total = __ldg(sum);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float4* input4 = (float4 *)input;
    int n4 = N/4;
     
    float4 data;
    float dat;
    for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
        float4 data = input4[idx];
        data.x = __fdividef(data.x,total);
        data.y = __fdividef(data.y,total);
        data.z = __fdividef(data.z,total);
        data.w = __fdividef(data.w,total);
        input4[idx] = data;       
    }
    int tail = n4*4;
    for(int idx=tail+index;idx<N;idx+=gridDim.x*blockDim.x){
        dat = __fdividef(input[idx],total);
        input[idx] = dat;
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 4 * 108;
    float *sum, *maxmum, *partial;
    cudaMalloc(&partial,sizeof(float)*blocksPerGrid);
    cudaMalloc(&maxmum,sizeof(float));
    cudaMemset(maxmum,-INFINITY,sizeof(float));
    cudaMalloc(&sum,sizeof(float));
    cudaMemset(sum,0,sizeof(float));

    softmax_partialmax<<<blocksPerGrid, threadsPerBlock>>>(input, N, partial);
    softmax_max<<<blocksPerGrid, threadsPerBlock>>>(partial, blocksPerGrid, maxmum);
    softmax_reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, sum, maxmum);
    softmax_norm<<<blocksPerGrid, threadsPerBlock>>>(output, N, sum);
    cudaDeviceSynchronize();
}
