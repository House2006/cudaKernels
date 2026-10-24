#include <cuda_runtime.h>
#include <stddef.h>

__global__ void rope_kernel(const float* input, float* output,
                            const long long* position, const float* sinCos,
                            int tokens, int heads, int headSize,
                            int rotaryDim, int maxPos, size_t totalWork){
    int halfDim = rotaryDim / 2;
    int workPerHead = headSize - halfDim;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t idx = index; idx < totalWork; idx += stride){
        size_t headIndex = idx / workPerHead;
        int feature = idx % workPerHead;
        int token = headIndex / heads;
        int pos = (int)position[token];
        size_t base = headIndex * headSize;

        if(feature < halfDim){
            int first = feature;
            int second = feature + halfDim;
            if(pos >= 0 && pos < maxPos){
                size_t sinCosBase = (size_t)pos * rotaryDim;
                float sin = sinCos[sinCosBase + first];
                float cos = sinCos[sinCosBase + halfDim + first];
                float x1 = input[base + first];
                float x2 = input[base + second];
                output[base + first] = x1 * cos - x2 * sin;
                output[base + second] = x2 * cos + x1 * sin;
            }else{
                output[base + first] = input[base + first];
                output[base + second] = input[base + second];
            }
        }else{
            int tailIndex = rotaryDim + feature - halfDim;
            output[base + tailIndex] = input[base + tailIndex];
        }
    }
}

void launch_rope(const float* input, float* output, const long long* position,
                 const float* sinCos, int tokens, int heads, int headSize,
                 int rotaryDim, int maxPos){
    int halfDim = rotaryDim / 2;
    int workPerHead = headSize - halfDim;
    size_t totalWork = (size_t)tokens * heads * workPerHead;

    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((totalWork - 1) / threadsPerBlock + 1);
    if(blocksPerGrid > 4 * 108){
        blocksPerGrid = 4 * 108;
    }

    rope_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, position, sinCos,
                                                    tokens, heads, headSize,
                                                    rotaryDim, maxPos, totalWork);
}

// 输入按 [tokens, heads, headSize] 连续存储
// sinCos 按 [maxPos, rotaryDim] 存储，每行先放sin，再放cos
// queryHeads和keyHeads可以不同；position每个token对应一个位置
extern "C" void solve(const float* query, const float* key,
                      const long long* position, const float* sinCos,
                      float* queryOut, float* keyOut,
                      int tokens, int queryHeads, int keyHeads,
                      int headSize, int rotaryDim, int maxPos){
    if(tokens <= 0 || queryHeads <= 0 || keyHeads <= 0 || headSize <= 0 ||
       rotaryDim < 0 || rotaryDim > headSize || rotaryDim % 2 != 0 || maxPos <= 0){
        return;
    }

    launch_rope(query, queryOut, position, sinCos, tokens, queryHeads,
                headSize, rotaryDim, maxPos);
    launch_rope(key, keyOut, position, sinCos, tokens, keyHeads,
                headSize, rotaryDim, maxPos);
    cudaDeviceSynchronize();
}
