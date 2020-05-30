<!--
 * @Author: xieydd
 * @since: 2020-05-08 10:36:58
 * @lastTime: 2020-05-30 17:12:45
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
conv0            conv1  # vector<float> conv0_next_scale, 假设 a; a[2]=a[3]=conv2_scale_in  a[1]=conv4_scale_in ... 直到vector填满为止
                        # vector<float> conv1_next_scale, 假设 a1; Notice 这里要注意和下面的不一样 a1[4] = a1[5] = conv2_scale_in ; a1[6] = conv4_scale_in
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

4. 当 conv 后不是 split 而是跟 binaryop 后跟 split
conv    |
    binaryop
    split
conv    |    
5. Conv/convdw/inner product 后无 output_scale 直接输出, 这种情况暂时先做 float 输出
6. Average Pool 的计算溢出问题，int32 作为暂存值，不会出现溢出问题
```
2. Prepared work
```s
E = (scale_input_next_float / (scale_weight_float * scale_input_float))  * 2^F  
# cx: A_max 映射成 16bit 最大值
# We want to storage E before runtime, and F must be sured in runitime, so in there we use a fake F, in runtime will be recovery, but it is not best.
# We can get F in KLD algrithm, when get thread will get F
# Above think is wrong
# Just find max(scale_input_next_float / (scale_weight_float * scale_input_float)), and set BitN, we use 8, and get F
int8 = (input_int8 * weight_int8 + bias_int32) * E >>F 
#if use int8 bias
#bias_int8 = float2int8(bias_float*scale_next_float) # Do it in ncnn2int8
#int8 = (input_int8 * weight_int8)*E >> F + bias_int8
```

Notice: 1 and 2 will implement in tools/quantization/ncnn2int8.cpp

3. Changed Operation
```s
1. x86/convolution_x86.cpp              Done
2. x86/convolutiondepthwise_x86.cpp     Done
3. src/Mat.cpp                          Done
4. src/layer/concat.cpp                 Done
5. src/layer/permute.cpp                Done
6. src/layer/relu.cpp                   Done
7. src/layer/reshape.cpp                Done
8. src/layer/slice.cpp                  Done
9. src/layer/softmax.cpp                Can`t Do
10. src/layer/binaryop.cpp              Done
11. src/layer/clip.cpp                  Done
12. src/layer/flatten.cpp               Done    
13. src/layer/innerproduct.cpp          Done
14. src/layer/input.cpp                 Done
15. src/layer/padding.cpp               Done
16. src/layer/pooling.cpp               Done
17. src/layer/quantize.cpp              Doesn`t need
18. x86/convolution_sgemm_int8.h        Done
19. x86/convolutiondepthwise_3x3_int8.h Done
20. x86/convolutiondepthwise_x86.cpp    Done
21. src/layer/split.cpp                 Doesn`t need 
22. src/layer/shufflechannel.cpp        Doesn`t need  

# For Input layer need change float input to int8
# For Activation_type, shufflenetv2 use relu(type=1) and mbv2 use relu6(type=3) max/scale << right_shift
```

4. Runtime 
```s
1. Mobilenetv2 model have relu6, so max number need change with top_scale, for simple use, before runtime storage max*top_scale,  when see mbv2 param, with relu6 activation , top_scale only have one number, so it can storage in param
先只存 top_scale 更大的 6*top_scale_larger, 对 6*top_scale_larger/top_scale_small


2. When Split have different top_scales in mbv2
崔鑫做法是取先获取scale大的对应的cube， 再通过该cube计算出scale小的cube。 http://gitlab-iot.yzs.io/cuixin/ncnn_conv_std/blob/mbv2/src/layer/x86/convolution_x86.cpp#L497

3. BinaryOp 溢出问题
崔鑫是通过拓展到 int32 后，直接上下裁剪, 直接裁剪是否太暴力 http://gitlab-iot.yzs.io/cuixin/ncnn_conv_std/blob/mbv2/src/mat.cpp#L577
经过分析可知， Binaryop 的和到下一层，对下一层的输入的 scale 已经是在 127 限制下生成的
例如 a_float[] = [1.0,0.5]
    b_float[] = [1.0,1.27]
  和c_float[] = [2.0,1.77] c_int[] = [127, 113] c_bottom_scale=63.5
  假设 a_top_scale = c_bottom_scale
       b_top_scale = 100
    
  c_simulate_int = [64 + 100*63.5/100, 32+127*63.5/100]
                 = [127.5, 113]
  c_int[]        = [127, 113]

4. BinaryOp 两边量纲不一样，可能第一层一样但是第二层不一样
Split 分别保存 left_scale 和 right_scale
int8*right_scale/left_scale 同样做提前保存和移位操作
```

mbv2 result
|测试方案	|cosine distance|	cosine angle	|像素点	|相对误差 （MRE）
|  ----  | ----  | ---- | ---- | ----
|原始 ncnn float 和 mbv2 全量化（全量化方案一）|	0.9998227	|0.72667	| 6.616	|0.063（过滤掉 mre>1 的情况）,不过滤是 0.092
|原始 ncnn float 和崔鑫全量化|	0.9997271|	0.996|	 7.97|	0.17
|原始 ncnn float 和 半量化|	0.9999787|	0.2637	|2.95	|0.023

2020-05-26 complete mbv2 int8 quantization



2020-05-27~29
out
|测试方案	|cosine distance|	cosine angle	|绝对误差（MAE）|相对误差 （MRE）
|  ----  | ----  | ---- | ---- | ----
|原始 ncnn float 和 int32整体移位量化|	0.9776	|12.02|0.208|0.32
|原始 ncnn float 和 半量化|	0.99567|	5.2599	|0.088|0.178
845
|测试方案	|cosine distance|	cosine angle	|绝对误差(MAE)|相对误差 （MRE）
|  ----  | ----  | ---- | ---- | ----
|原始 ncnn float 和 int32整体移位量化|	0.99990	|	0.6383|0.0008|0.23
|原始 ncnn float 和 半量化|	0.99995|	0.4399	|0.00049|0.069


2020-05-30 start full int8 quantization of shufflenetv2
1. Clip not changed compared with mbv2, because shufflenetv2 not use clip
2. conv and convdw detele top_scale_num , top_scale_num is always 2(top_scale_int and shift), innerproduct not use, don`t change
3. change slice: add smallest_top_scale/real_top_scale( int and N_shift)
4. change split :
4. main change of ncnn2int8
    1. find all top_scale, and smallest_top_scale/real_top_scale only for slice and split
    2. storage it