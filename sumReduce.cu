
//block级规约，线程通过共享内存和同步协作



__global__ void sumReduce_v0 (const int *input,int *output,int N){
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (index<N)?input[index]:0;
    __syncthreads();

    for(int step = 1; step<blockDim.x; step *= 2){
        if(tid%(2*step)==0){
            smem[tid] += smem[tid+step];
        }
        __syncthreads();
    }

    if(tid==0){
        output[blockIdx.x] = smem[0];
    }

}
//每个warp里能够并行执行的thread指数级下滑，造成带宽和算力浪费，整体吞吐量下降
//考虑每次折半后让活跃线程尽可能集中


__global__ void sumReduce_v2 (const int *input,int *output,int N){
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (index<N)?input[index]:0;
    __syncthreads();

    for(int step = 1; step<blockDim.x; step *= 2){
        if(tid<blockDim.x/(step*2)){
            smem[tid*step] += smem[tid*step+step];
        }
        __syncthreads();
    }

    if(tid==0){
        output[blockIdx.x] = smem[0];
    }

}

//每轮循环线程重新映射负责的数据，将线程与他负责处理的数据的位置解绑，可以让活跃线程始终保持在前面的warp
//一个warp里不同thread要访问的数据不相邻且间隔始终是2的幂次，会造成bank conflict



__global__ void sumReduce_v3 (const int *input,int *output,int N){
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (index<N)?input[index]:0;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(tid<stride){
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }

    if(tid==0){
        output[blockIdx.x] = smem[0];
    }

}

//规约间隔从远到近，活跃线程始终是连续的低编号线程，线程集中且访问的数据相邻
//实际上最后一个warp不需要__syncthreads(),可以让后log32轮没有同步等待开销

__global__ void sumReduce_v4 (const int *input,int *output,int N){
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (index<N)?input[index]:0;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 32; stride /= 2){
        if(tid<stride){     //条件内置，所有warp只能执行全部循环判断
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }

    if(tid<32){             //外置活跃线程的选择判断，其他warp就不需要循环判断了：if<for
        for(int stride = 32; stride>0; stride /= 2){       
            smem[tid] += smem[tid+stride];        
        }   
    }
    

    if(tid==0){
        output[blockIdx.x] = smem[0];
    }

}

//循环由于每轮都需要条件判断和指令跳转，造成每个warp执行时流水线被打断，并且需要额外开销
//顺序执行 <= if(分歧线程无法参与并行造成浪费）< for 指令缓存失效 < if...else...（两个代码块串行） 

__global__ void sumReduce_v5 (const int *input,int *output,int N){
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (index<N)?input[index]:0;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 32; stride /= 2){
        if(tid<stride){
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }

    // 最后 Warp 内展开,用顺序执行替换for循环
    if (tid < 32) {
        volatile int* vsmem = smem; //使用vsmem会强制编译器写回smem，否则其他线程将读到上一轮未更新旧值
        //warp内指令级同步，满足时序逻辑：每个线程读到的都是原来的值，不会让一个线程读取本轮已经更新过的值
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }   
    

    if(tid==0){
        output[blockIdx.x] = smem[0];
    }

}

//warp级规约：warp shuffle

__device__ int warpReduce(int val){
    for(int offset = 16;offset > 0;offset /= 2){
        val += __shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}
__global__ void sumReduce_v6(const int *input,int *output,int N){
    extern __shared__ int smem[32];
    int tid = threadIdx.x;
    int wid = tid/32;
    int lane = tid%32;
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int val = index<N?input[index]:0;
    val = warpReduce(val);
    if(lane == 0){
        smem[wid] = val;
    }
    __syncthreads();

    int warpNums = blockDim.x/32;
    if(wid == 0){
        val = lane<warpNums?smem[lane]:0;
        val = warpReduce(val);
    }

    if(tid==0) output[blockIdx.x] = val;

}

__global__ void sumReduce_v7(const float *input,float *output,int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid%32;
    int wid = tid/32;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    int n4 = N/4;

    float val = 0.0f;
    float4 data;
    for(int idx=index;idx<n4;idx+=gridDim.x*blockDim.x){
        float4 data = input4[idx];
        val += data.x+data.y+data.z+data.w;
    }
    int tail = n4*4;
    for(int idx=tail+index;idx<N;idx+=gridDim.x*blockDim.x){
        val += input[idx];
    }

    __shared__ float warpSums[32];
    val = warpReduce(val);
    if(lane == 0){
        warpSums[wid] = val;
    }
    __syncthreads();

    int warpNums = blockDim.x/32;
    if(wid == 0){
        val = lane<warpNums?warpSums[lane]:0;
        val = warpReduce(val);
    }

    if(tid==0) output[blockIdx.x] = val;
}



 
