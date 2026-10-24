#include "cuda_header.cuh"


__global__ void saxpy (const float* x, float* y,float alpha,int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<N;i+=stride){
        y[i] += alpha * x[i];
    }
     
}

float* saxpyCPU(const float *x,const float *y,float alpha,int N){
    float *ans = (float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        ans[i] = x[i]*alpha + y[i];
    }
    return ans;
}
int main (int argc,char** argv){
    if(argc!=3){
        printf("Usage: %s <vector size> <alpha>\n", argv[0]);
        return -1;
    }   

    long long parsed = atoll(argv[1]);
    if(parsed <= 0 || parsed > INT_MAX){
        printf("Error: vector size must be a positive integer less than or equal to %d\n", INT_MAX);
        return -1;
    }

    int N = atoi(argv[1]);
    int size = N * sizeof(float);
    float alpha = atof(argv[2]);
    
    float *x_h,*y_h,*ans_gpu;
    cudaMallocHost(&x_h,size);
    cudaMallocHost(&y_h,size);
    cudaMallocHost(&ans_gpu,size);
    for(int i=0;i<N;i++){
        x_h[i] = i;
        y_h[i] = 2*i;
    }

    float *x_d,*y_d;
    cudaMalloc(&x_d,size);
    cudaMalloc(&y_d,size);
    cudaMemcpy(x_d,x_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,y_h,size,cudaMemcpyHostToDevice);

    int blockSize = 128;    
    int gridSize_max = (N+blockSize-1)/blockSize;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int gridSize = std::min(gridSize_max,prop.multiProcessorCount * 4);

    saxpy<<<gridSize,blockSize>>>(x_d,y_d,alpha,N);
    cudaMemcpy(ans_gpu,y_d,size,cudaMemcpyDeviceToHost);//CPU端也会阻塞，因为涉及主存数据更新

    float *ans_cpu = saxpyCPU(x_h,y_h,alpha,N);
    int i;
    for(i=0;i<N;i++){
        if(ans_cpu[i]!=ans_gpu[i])  break;
    }
    printf("%s",N-i?"Failure\n":"Success\n");
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFreeHost(x_h);
    cudaFreeHost(y_h);
    cudaFreeHost(ans_gpu);   
    return 0;   


}