#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    /*
运算可以包含属性，属性值在向计算图中添加运算时设置。这些值用于配置运算，用户可以在内核实现中以及运算注册的输入和输出类型中访问它们的值。特性是常量，必须在计算图构造时定义。
*/
    // .Attr("preserve_index: {'0','1','2','3'}")
    //{'<string1>', '<string2>'}：值必须是值为 <string1> 或 <string2> 的字符串
    .Attr("preserve_index: int >= 0 = 1")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

/*
ZeroOut 运算会将一个包含 32 位整数的张量 to_zero 作为输入，并输出一个包含 32 位整数的张量 zeroed。该运算还使用形状函数来确保输出张量与输入张量的形状相同。例如，如果输入是形状为 [10, 20] 的张量，则此形状函数会指定输出形状也是 [10, 20]。

注：运算名称必须采用驼峰命名法，并且对于在二进制文件中注册的所有其他运算，该名称必须唯一。
*/

/*
实现运算的内核
在定义接口后，您需要为运算提供一个或多个实现。要创建其中一个内核，请先创建一个扩展 OpKernel 并重写 OpKernel 方法的类。Compute 方法提供了一个类型为 OpKernelContext* 的 context 参数，您可以从中访问输入张量和输出张量等有用信息。
*/
class ZeroOutOp : public OpKernel
{
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // Get the index of the value to preserve
        OP_REQUIRES_OK(context,
                       context->GetAttr("preserve_index", &preserve_index_));
        // Check that preserve_index is positive
        OP_REQUIRES(context, preserve_index_ >= 0,
                    errors::InvalidArgument("Need preserve_index >= 0, got ",
                                            preserve_index_));
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()), errors::InvalidArgument("ZeroOut expects a 1-D vector."));
        auto input = input_tensor.flat<int32>();
        OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                    errors::InvalidArgument("preserve_index out of range"));
        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        auto output_flat = output_tensor->flat<int32>();

        const int N = input.size();
        for (int i = 1; i < N; i++)
        {
            output_flat(i) = 0;
        }
        output_flat(preserve_index_) = input(preserve_index_);
    }

private:
    int preserve_index_;
};

/*
实现内核后，您需要将其注册到 TensorFlow 系统。在注册中，您要指定此内核将在哪些不同约束下运行。例如，您可能有一个面向 CPU 的内核，以及一个面向 GPU 的内核。

要针对 ZeroOut 运算执行此操作，请将以下代码添加到 zero_out.cc 中：
*/

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);