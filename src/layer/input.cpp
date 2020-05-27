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

#include "input.h"
#include "quantize.h"
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

DEFINE_LAYER_CREATOR(Input)

Input::Input()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = false;
}

int Input::load_param(const ParamDict &pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);
    use_int8_inference = pd.get(8, 0);

    return 0;
}

int Input::load_model(const ModelBin &mb)
{
    if (use_int8_inference)
    {
        Mat scales = mb.load(1, 1);
        int *ptr = scales.channel(0);
        scale = ptr[0];
        Mat shift = mb.load(1, 1);
        const int *shift_ptr = shift.channel(0);
        right_shift = shift_ptr[0];
    }
    return 0;
}

// int Input::forward_inplace(Mat & /*bottom_top_blob*/, const Option & /*opt*/) const
// {
//     fprintf(stdout, "xxxxxx\n");
//     fflush(stdout);
//     return 0;
// }

int Input::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    if (use_int8_inference)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
    // Mat bottom_blob_int = bottom_blob;
    // if (!support_inplace)
    //     return -1;

    // top_blob = bottom_blob.clone(opt.blob_allocator);
    // if (top_blob.empty())
    //     return -100;
    top_blob = bottom_blob;

    return 0;
}

int Input::forward_int8(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    top_blob.create(w, h, channels, 1u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    for (int q = 0; q < channels; q++)
    {
        const float *ptr = bottom_blob.channel(q);
        signed char *outptr = top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (right_shift < 0)
            {
                //outptr[i] = float2int8(int(ptr[i] * scale + 0.5) >> (-right_shift));
                outptr[i] = float2int8(int(ptr[i] * scale + 0.5) >> (-right_shift));
            }
            else
            {
                outptr[i] = float2int8(int(ptr[i] * scale + 0.5) << (right_shift));
            }
        }
    }
    int len = top_blob.c * top_blob.w * top_blob.h;
    std::vector<signed char> result;
    result.resize(len);
    mat2vector_signed_char(top_blob, result);
    return 0;
}

} // namespace ncnn
