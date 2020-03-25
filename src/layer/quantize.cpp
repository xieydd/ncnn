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

#include "quantize.h"

#include <math.h>

namespace ncnn
{

DEFINE_LAYER_CREATOR(Quantize)

Quantize::Quantize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Quantize::load_param(const ParamDict &pd)
{
    // TODO change scale to get(0,1.f) and quantize.h scale to float
    scale = pd.get(0, 1);
    position_bottom_scale = pd.get(1, 5);
    position_scale_in = pd.get(2, 18);

    return 0;
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return (signed char)int32;
}

static inline signed char int2int8(int int32)
{
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return (signed char)int32;
}

static inline int32_t float2int32(float v)
{
    int32_t int32 = static_cast<int32_t>(round(v));
    return int32;
}

// This function will cause bigger error than function above
//#include <stdint.h>
// static inline int32_t float2int32(float x)
// {
//     //取得符号位，设置掩码
//     uint32_t n = ((*(uint32_t *)&x) & 0x80000000) ? 0xFFC00000 : 0; //一个三元操作符，直接储存掩码
//     x += 12582912.0f;                                               //魔法数字加法
//     return ((*(uint32_t *)&x) & 0x3FFFFF) | n;                      //直接or运算
// }

int Quantize::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        if (opt.use_int32_storage)
        {
            top_blob.create(w, (size_t)4u, opt.blob_allocator);
        }
        else
        {
            top_blob.create(w, (size_t)1u, opt.blob_allocator);
        }

        if (top_blob.empty())
            return -100;

        if (opt.use_int32_storage)
        {
            const float *ptr = bottom_blob;
            int32_t *outptr = top_blob;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int32(ptr[i] * scale);
            }
        }
        else
        {
            signed char *outptr = top_blob;
            if (opt.use_int_internal)
            {
                const int *ptr = bottom_blob;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = int2int8((int(ptr[i] * scale) >> position_bottom_scale) >> position_scale_in);
                }
            }
            else
            {
                const float *ptr = bottom_blob;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = float2int8(ptr[i] * scale);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        signed char *outptr = top_blob;

        if (opt.use_int_internal)
        {
            const int *ptr = bottom_blob;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                outptr[i] = int2int8((int(ptr[i] * scale) >> position_bottom_scale) >> position_scale_in);
            }
        }
        else
        {
            const float *ptr = bottom_blob;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (opt.use_int_internal)
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int *ptr = bottom_blob.channel(q);
                signed char *outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = int2int8((int(ptr[i] * scale) >> position_bottom_scale) >> position_scale_in);
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float *ptr = bottom_blob.channel(q);
                signed char *outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float2int8(ptr[i] * scale);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
