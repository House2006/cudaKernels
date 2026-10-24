#include "cuda_header.cuh"

#define WARM_UP 10
#define REPEAT 5

__global__ void  vectorMulReduce(const float* A,const float* B,float* partial,int N){

    int tid = threadIdx.x;  //该线程负责的共享内存数据的位置
    int initIndex = blockIdx.x*blockDim.x + threadIdx.x; //该线程负责的全局显存数据的位置
    int gridStride = blockDim.x * gridDim.x; 
    extern __shared__ float ans[]; //block内部线程间合作
    float threadSum = 0.0f;

    for(int index=initIndex;index<N;index+=gridStride){
        threadSum += A[index] * B[index];
    }
    ans[tid] = threadSum;
    __syncthreads();  //保证跨warp线程协作时的数据一致性。要注意避免某个block的thread一部分能运行到，另一部分运行不到

    for(int stride = blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride)
            ans[tid] += ans[tid+stride];   
        __syncthreads();
    }    


    /*
    for(int index=initIndex;index<N;index+=gridStride){
        ans[tid] = index<N?A[index]*B[index]:0;  //avoid processing the threads that are out of bounds
        __syncthreads();                //prepare to cooperate with other threads in one block

        for(int stride = blockDim.x/2;stride>0;stride>>=1){
            if(tid<stride)
                ans[tid] += ans[tid+stride];   
            __syncthreads();
        }

        if(threadIdx.x==0) {        //暂存到共享内存，减少对全局内存的原子加法
            blockSum += ans[0];
        }   
    }
    */
    if(threadIdx.x==0){
        atomicAdd(partial,ans[0]);
    }

    
}

void vectorMulCPU(float* A,float* B,float* c,int N){
    *c = 0;
    for(int i=0;i<N;i++){
        *c += A[i]*B[i];
    }
}

int main(int argc,char** argv){

    if(argc!=2){
        printf("Usage: %s <vector size>\n", argv[0]);
        return -1;
    }   

    long long parsed = atoll(argv[1]);
    if(parsed <= 0 || parsed > INT_MAX){
        printf("Error: vector size must be a positive integer less than or equal to %d\n", INT_MAX);
        return -1;
    }
   
   
    //========向量初始化========
    int N = atoi(argv[1]);
    int size = N*sizeof(float);    
    float  *h_A, *h_B;
    cudaMallocHost(&h_A,size);
    cudaMallocHost(&h_B,size);
    float  h_ans = 0;
    for(int i=0;i<N;i++){
        h_A[i] = (float)i;
        h_B[i] = (float)(i*2);
    }

    // ========== GPU计算 ========== 
    float *d_A,*d_B,*d_ans;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_ans,sizeof(float));
    cudaMemset(d_A,0,size);
    cudaMemset(d_B,0,size); 
    cudaMemset(d_ans,0,sizeof(float));

    //GPU warm-up
    for(int i=0;i<WARM_UP;i++){
        vectorMulReduce<<<(N+127)/128,128,128*sizeof(float)>>>(d_A,d_B,d_ans,N);
    }
    cudaDeviceSynchronize();

    int blockSize = 128;    
    int gridSize_max = (N+blockSize-1)/blockSize;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int gridSize = std::min(gridSize_max,prop.multiProcessorCount * 4);



    float gpu_total_ms , kernel_ms ;
    float gpu_total_duration = 0,kernel_duration = 0;
    for(int i=0;i<REPEAT;i++){
        cudaEvent_t  start , stop ,kernel_start ,kernel_stop;     
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);

        cudaEventRecord(start);
        cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

        cudaEventRecord(kernel_start);
        vectorMulReduce<<<gridSize,blockSize,blockSize*sizeof(float)>>>(d_A,d_B,d_ans,N);
        cudaEventRecord(kernel_stop);   

        cudaMemcpy(&h_ans,d_ans,sizeof(float),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&gpu_total_ms, start, stop);
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
        gpu_total_duration += gpu_total_ms;
        kernel_duration += kernel_ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);

        cudaMemset(d_ans,0,sizeof(float));
    }
    gpu_total_ms = gpu_total_duration/REPEAT;
    kernel_ms = kernel_duration/REPEAT;
    

    
    // ========== CPU计算 ==========
    float ans_cpu = 0;
    float cpu_duration = 0;
    for(int i=0;i<REPEAT;i++){
        auto cpu_start = std::chrono::high_resolution_clock::now();
        vectorMulCPU(h_A, h_B, &ans_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_duration += std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    }
    float cpu_ms = cpu_duration / REPEAT;
    
    
    //==========结果验证=========
    if (fabs(h_ans - ans_cpu) > ans_cpu*1e-3 + 1e-5) {      //低于1e-5时不满足相对误差1e-4的条件也判等
        printf("Results do not match! GPU: %f, CPU: %f\n", h_ans, ans_cpu); 
    } else {
        printf("Results match! Result: %f\n", h_ans);
    }
    
    //==========性能计算=========   
    
    printf("CPU time: %.4f ms\n", cpu_ms);
    printf("GPU kernel time: %.4f ms\n", kernel_ms);
    printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Speedup (end-to-end): %.1fx\n", cpu_ms / gpu_total_ms);

    float bandwidth_gb = (2.0f * size + gridSize *sizeof(float)) / (kernel_ms * 1e-3f) / 1e9f;
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gb);

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ans);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
   

    return 0;


}



