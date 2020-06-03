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

#include "concat.h"
#include <algorithm>
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

DEFINE_LAYER_CREATOR(Concat)

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict &pd)
{
    axis = pd.get(0, 0);
    use_int8_inference = pd.get(8, 0);

    return 0;
}

int Concat::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    if (use_int8_inference)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;

    if (dims == 1) // axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float *outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const float *ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float *outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];

            int size = w * bottom_blob.h;

            const float *ptr = bottom_blob;
            memcpy(outptr, ptr, size * elemsize);

            outptr += size;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float *outptr = top_blob.row(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                const float *ptr = bottom_blob.row(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.c;
            size_t size = bottom_blob.cstep * channels;

            const float *ptr = bottom_blob;
            float *outptr = top_blob.channel(q);
            memcpy(outptr, ptr, size * elemsize);

            q += channels;
        }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float *outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const float *ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float *outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat &bottom_blob = bottom_blobs[b];

                    const float *ptr = bottom_blob.channel(q).row(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Concat::forward_int8(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;

    if (dims == 1) // axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        signed char *outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const int *ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat &top_blob = top_blobs[0];
        std::string type_name = "output";
        std::string softmax_name = "844";
        if (name == type_name || name == softmax_name)
        {
            top_blob.create(w, top_h, 4u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            float *outptr = top_blob;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int size = w * bottom_blob.h;

                const signed char *ptr = bottom_blob;
                for (int i = 0; i < size; i++)
                {
                    if (name == type_name)
                    {
                        outptr[i] = 1.0f * ptr[i] / 7.0;
                    }
                    else
                    {
                        outptr[i] = 1.0f * ptr[i] / 10.0;
                    }
                }
                outptr += size;
            }
        }
        else
        {
            top_blob.create(w, top_h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            signed char *outptr = top_blob;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int size = w * bottom_blob.h;

                const signed char *ptr = bottom_blob;
                memcpy(outptr, ptr, size * elemsize);

                outptr += size;
            }
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            signed char *outptr = top_blob.row_signed_char(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                const signed char *ptr = bottom_blob.row_signed_char(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat &top_blob = top_blobs[0];

        if (name == "692")
        {
            top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels = 0;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int size = w * bottom_blob.h;

                for (int c = 0; c < bottom_blob.c; c++)
                {
                    const signed char *ptr = bottom_blob.channel(c);
                    signed char *outptr = top_blob.channel(c + channels);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = float2int8(1.0f * ptr[i] * 18.017279 / 21.516098);
                    }
                    outptr += size;
                }
                channels += bottom_blob.c;
            }
        }
        else
        {
            top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int q = 0;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int channels = bottom_blob.c;
                size_t size = bottom_blob.cstep * channels;

                const signed char *ptr = bottom_blob;
                signed char *outptr = top_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                q += channels;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char *outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat &bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const signed char *ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat &bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat &top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char *outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat &bottom_blob = bottom_blobs[b];

                    const signed char *ptr = bottom_blob.channel(q).row_signed_char(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
