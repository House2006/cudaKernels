# CUDA 算子练习

这是一个用于练习 CUDA 编程的个人项目，包含 LeetGPU 相关练习，以及自己实现和优化的一些常见 GPU 算子。学习重点放在线程与线程间的协作关系（warp/block/grid），线程与数据间的对应关系（register/shared memory/global memory）。

## 内容

| 文件 | 内容 |
| --- | --- |
| `vectorAdd.cu` | 向量加法示例，包含 CPU 对照、正确性检查和计时；另有向量化加载实现 |
| `vectorMul.cu` | 向量点积，使用 block 内共享内存规约并累加结果 |
| `saxpy.cu` | SAXPY：`y[i] += alpha * x[i]`，包含 CPU 对照 |
| `sumReduce.cu` | 整数/浮点求和规约的多个版本，演示共享内存、warp 规约等实现思路 |
| `softmax.cu` | Softmax，分阶段计算最大值、指数和及归一化 |
| `onlineSoftmax.cu` | 使用在线 `(max, sum)` 统计量的 Softmax 实现 |
| `swiGLU.cu` | SwiGLU 激活算子，包含 SiLU 与逐元素乘法 |
| `gemm.cu` | 使用共享内存分块的矩阵乘法（GEMM） |
| `rope.cu` | RoPE（旋转位置编码），支持 query 和 key 使用不同 head 数 |
| `cuda_header.cuh` | CUDA 错误检查宏及常用头文件 |
| `test.cu` | 最小 CUDA kernel 启动示例 |
| `Makefile` | 示例程序和算子对象文件的构建规则 |

## 环境要求

- 支持 CUDA 的 NVIDIA GPU（运行 GPU 程序时需要）
- 已安装 CUDA Toolkit，且 `nvcc` 可用
- GNU Make

## 构建

在项目根目录运行：

```bash
make
```

`make` 默认构建所有目标：

- 示例可执行文件放在 `build/bin/`
- 算子编译出的对象文件放在 `build/obj/`

也可以分别构建：

```bash
make demos   # 构建可运行示例
make kernels # 将算子编译为 .o 文件
make clean   # 删除 build/ 目录
```

可通过变量指定编译器和编译选项，例如：

```bash
make NVCC=/path/to/nvcc NVCCFLAGS="-O2"
```

## 运行示例

```bash
./build/bin/vectorAdd 1048576
./build/bin/vectorMul 1048576
./build/bin/saxpy 1048576 2.0
./build/bin/test
```

`vectorAdd` 会进行结果校验并打印 CPU、GPU kernel 和端到端耗时；`vectorMul` 和 `saxpy` 也会与 CPU 结果进行对照。运行性能数据会受 GPU 型号、驱动、CUDA 版本和系统负载影响。

上述命令可以直接运行并看到终端输出。`make kernels` 目前只编译各算子为对象文件，没有为这些 `solve` 接口提供独立的测试程序；要运行它们，需要由调用方分配设备内存并调用对应接口。

## 算子接口

`softmax.cu`、`onlineSoftmax.cu`、`swiGLU.cu`、`gemm.cu` 和 `rope.cu` 提供了名为 `solve` 的主机端入口，参数中的输入输出指针预期指向 GPU 设备内存。它们分别实现 Softmax、在线 Softmax、SwiGLU、GEMM 和 RoPE。每个文件可以单独编译；由于多个文件都定义了 `solve`，不要将这些对象文件直接链接进同一个可执行文件。

`sumReduce.cu` 则主要用于比较多种规约 kernel 的实现方式。它不提供统一的 `solve` 包装入口。

## 说明

这是持续练习中的代码仓库，算子实现和性能表现可能会随着学习继续调整。部分文件是独立的练习或接口实现，不构成一个统一的库。
