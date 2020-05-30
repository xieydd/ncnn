/*
 * @Author: xieydd
 * @since: 2020-05-08 10:35:56
 * @lastTime: 2020-05-30 19:52:56
 * @LastAuthor: Do not edit
 * @message: 
 */
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "split.h"

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return (signed char)int32;
}
namespace ncnn
{

DEFINE_LAYER_CREATOR(Split)

Split::Split()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = true;
}

int Split::load_param(const ParamDict &pd)
{
    use_int8_inference = pd.get(8, 0);
    use_factor = pd.get(9, 0);
    return 0;
}

int Split::load_model(const ModelBin &mb)
{
    if (use_factor)
        factors_mat = mb.load(2, 1);
    return 0;
}

int Split::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    if (use_int8_inference)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
    const Mat &bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}

int Split::forward_int8(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    if (use_factor)
    {
        std::vector<int> factors;
        factors.resize(2);
        mat2vector(factors_mat, factors);
        int dst = factors[0];
        int right_shift = factors[1];
        const Mat &bottom_blob = bottom_blobs[0];

        for (int i = 0; i < bottom_blobs.size(); i++)
        {
            Mat a = bottom_blobs[0].clone();
            for (int c = 0; c < bottom_blob.c; c++)
            {
                const signed char *input = bottom_blob.channel(c);
                signed char *output = a.channel(c);
                for (int h = 0; h < bottom_blob.h; h++)
                {
                    for (int w = 0; w < bottom_blob.w; w++)
                    {
                        if (right_shift < 0)
                        {
                            output[w] = (input[w] * dst) >> (-right_shift);
                        }
                        else
                        {
                            output[w] = (input[w] * dst) << right_shift;
                        }
                    }
                    input += bottom_blob.w;
                    output += bottom_blob.w;
                }
            }
            top_blobs[i] = a;
        }
    }
    else
    {
        const Mat &bottom_blob = bottom_blobs[0];
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            top_blobs[i] = bottom_blob;
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Split::forward(const std::vector<VkMat> &bottom_blobs, std::vector<VkMat> &top_blobs, VkCompute & /*cmd*/, const Option & /*opt*/) const
{
    //     fprintf(stderr, "Split::forward %p\n", bottom_blobs[0].buffer());

    const VkMat &bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
