<!--
 * @Author: xieydd
 * @since: 2020-05-08 10:36:58
 * @lastTime: 2020-05-09 11:57:08
 * @LastAuthor: Do not edit
 * @message: 
 -->
#### 2020-05-08

1. Condition need Solve
```s
1. split 后面分别跟 conv/convdw/inner product， split 前面的 conv 的  scale_next 是统一的
   conv0 #conv0_next_scale = conv1_input_sccale = conv2_input_scale
   split
conv1  conv2 # two layer`s scale_input is same
2. split 后如果有一分支为 conv/convdw/inner product, 另一分支为空，两分支以 BinaryOp 相连接
    conv0   # conv0_next_scale0 = conv1_scale_input; conv0_next_scale1 = conv2_scale_input
    split
conv1   |
    binaryop
    split
conv2   |
2.1. split 后一分支是 conv/convdw/inner product， 另一个分支是 padding + pooling 可以融合到 2 中
    conv0  # conv0_next_scale0 = conv1_scale_input; conv0_next_scale1 = inner_product_input_scale
    split
padding   conv1 # conv1_next_scale0 = conv2_scale_input; conv1_next_scale1 = inner_product_input_scale 
pooling   split
    |  padding conv2
      concat
      flatten
      inner product
3. 见下图
conv0            conv1  # vector<float> conv0_next_scale, 假设 a0; a[2]=a[6]=a[3]=a[7]=conv2_scale_in  a[1]=a[5]=conv4_scale_in ... 直到vector填满为止
                        # vector<float> conv1_next_scale, 假设 a1;
        concat          # [0,1,2,3,4,5,6,7]
        shufflechannel  # [0 4 1 5 2 6 3 7]
        slice           # [0 4 1 5]  [2 6 3 7] 
|               conv2
|               convdw0
|               conv3
        concat
        shufflechannel  # [0 2 4 6 1 3 5 7]
        slice           # [0 2 4 6] [1 3 5 7]
|               conv4

4. Conv/convdw/inner product 后无 output_scale 直接输出
```
2. Prepared work
```s
E = (scale_input_next_float / (scale_weight_float * scale_input_float))  * 2^F  
# cx: A_max 映射成 16bit 最大值
# We want to storage E before runtime, and F must be sured in runitime, so in there we use a fake F, in runtime will be recovery, but it is not best.
# We can get F in KLD algrithm, when get thread will get F
# Above think is wrong
# Just find max(scale_input_next_float / (scale_weight_float * scale_input_float)), and set BitN, we use 8, and get F
int8 = (input_int8 * weight_int8 + bias_int32) * E >>F  cx: bias_int32 -> bias_int8
#if use int8 bias
bias_int8 = float2int8(bias_float*scale_next_float) # Do it in ncnn2int8
int8 = (input_int8 * weight_int8)*E >> F + bias_int8
```

Notice: 1 and 2 will implement in tools/quantization/ncnn2int8.cpp

3. Changed Operation
```s
1. x86/convolution_x86.cpp
2. x86/convolutiondepthwise_x86.cpp
3. src/Mat.cpp
4. 
```