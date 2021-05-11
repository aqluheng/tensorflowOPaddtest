# 请记住，生成的函数将获得一个蛇形名称（以符合 PEP8）。因此，如果您的运算在 C++ 文件中命名为 ZeroOut，则 Python 函数将称为 zero_out。
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
    print(zero_out_module.zero_out([1, 2, 3, 4]).eval())

# Prints
'''
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
这个约束数值类型,使用时用t=tf.int32这样的去满足
除了上述的外,还有些快捷方式:
    numbertype：类型 type 仅限于数值（非字符串和非布尔）类型。
    realnumbertype：类似于 numbertype，没有复杂类型。
    quantizedtype：类似于 numbertype ，但只是量化的数值类型。
'''

'''
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
'''
