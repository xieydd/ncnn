// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "clip.h"

#include <float.h>

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return (signed char)int32;
}

namespace ncnn
{

DEFINE_LAYER_CREATOR(Clip)

Clip::Clip()
{
    one_blob_only = true;
    support_inplace = true;
}

int Clip::load_param(const ParamDict &pd)
{
    min = pd.get(0, -FLT_MAX);
    max = pd.get(1, FLT_MAX);
    use_int8_inference = pd.get(8, 0);
    scales = pd.get(9, Mat());

    return 0;
}

int Clip::forward_inplace_int8(Mat &bottom_top_blob, const Option &opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    std::vector<int> scales_v;

    int scale_size = scales.w * scales.h * scales.c;
    scales_v.resize(scale_size);
    mat2vector(scales, scales_v);

    // TODO Not pre-channel
    int larger = scales_v[0];
    int right_shift = scales_v[1];
    if (scale_size == 3)
    {
        if (larger < scales_v[1])
        {
            larger = scales_v[1];
        }
        right_shift = scales_v[2];
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        signed char *ptr = bottom_top_blob.channel(q);
        int min_int8;
        int max_int8;
        if (right_shift < 0)
        {
            // TODO Only for relu6
            min_int8 = 0;
            max_int8 = static_cast<int>((int(max * larger) >> (-right_shift)) + 0.5) > 127 ? 127 : static_cast<int>((int(max * larger) >> (-right_shift)) + 0.5);
            //fprintf(stdout, "xxxxxx %d\n", max_int8);
        }
        else
        {
            min_int8 = 0;
            max_int8 = static_cast<int>((int(max * larger) << (right_shift)) + 0.5) > 127 ? 127 : static_cast<int>((int(max * larger) << (right_shift)) + 0.5);
        }

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < min_int8)
                ptr[i] = min_int8;
            if (ptr[i] > max_int8)
                ptr[i] = max_int8;
        }
    }

    return 0;
}

// int Clip::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
// {
//     if (bottom_top_blob.elemsize == 1u)
//     {
//         return Clip::forward_inplace_int8(bottom_top_blob, opt);
//     }

//     int w = bottom_top_blob.w;
//     int h = bottom_top_blob.h;
//     int channels = bottom_top_blob.c;
//     int size = w * h;

//     #pragma omp parallel for num_threads(opt.num_threads)
//     for (int q=0; q<channels; q++)
//     {
//         float* ptr = bottom_top_blob.channel(q);

//         for (int i=0; i<size; i++)
//         {
//             if (ptr[i] < min)
//                 ptr[i] = min;
//             if (ptr[i] > max)
//                 ptr[i] = max;
//         }
//     }

//     return 0;
// }

int Clip::forward_inplace(Mat &bottom_top_blob, const Option &opt) const
{
    if (use_int8_inference)
    {
        return forward_inplace_int8(bottom_top_blob, opt);
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float *ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < min)
                ptr[i] = min;
            if (ptr[i] > max)
                ptr[i] = max;
        }
    }

    return 0;
}

} // namespace ncnn
