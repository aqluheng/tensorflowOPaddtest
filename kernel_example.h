/*
多线程 CPU 内核
要编写多线程 CPU 内核，可以使用 work_sharder.h 中的 Shard 函数。此函数会在配置为用于运算内线程的线程之间对计算函数进行分片（请参见 config.proto 中的 intra_op_parallelism_threads）。
*/

/*
GPU 内核
GPU 内核分为两部分实现：OpKernel 内核和 CUDA 内核及其启动代码。

有时，OpKernel 实现在 CPU 和 GPU 内核之间很常见，例如检查输入和分配输出。在这种情况下，建议的实现是：

定义在设备上模板化的 OpKernel 和张量的基元类型。
要对输出进行实际计算，Compute 函数会调用模板化仿函数结构体。
针对 CPUDevice 的仿函数特殊版本在同一文件中定义，但针对 GPUDevice 的仿函数特殊版本在 .cu.cc 文件中定义，因为它将使用 CUDA 编译器进行编译。
*/
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
    void operator()(const Device& d,int size, const T*in, const T* out);
};

#ifdef GOOGLE_CUDA
template<typename T>
struct ExampleFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d,int size,const T* in, T* out);
}
#endif

#endif KERNEL_EXAMPLE_H_