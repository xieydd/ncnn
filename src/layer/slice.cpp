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

#include "slice.h"

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

DEFINE_LAYER_CREATOR(Slice)

Slice::Slice()
{
}

int Slice::load_param(const ParamDict &pd)
{
    slices = pd.get(0, Mat());
    axis = pd.get(1, 0);
    use_int8_inference = pd.get(8, 0);
    use_factor = pd.get(9, 0);

    return 0;
}

int Slice::load_model(const ModelBin &mb)
{
    if (use_factor)
        scales = mb.load(2, 1);
    return 0;
}

int Slice::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    if (use_int8_inference)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
    const Mat &bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    const int *slices_ptr = slices;

    if (dims == 1) // axis == 0
    {
        int w = bottom_blob.w;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float *ptr = (const float *)bottom_blob + q;
            float *outptr = top_blob;
            memcpy(outptr, ptr, slice * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((h - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = w * slice;

            const float *ptr = bottom_blob.row(q);
            float *outptr = top_blob;
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < h; j++)
            {
                float *outptr = top_blob.row(j);
                const float *ptr = bottom_blob.row(j) + q;
                memcpy(outptr, ptr, slice * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, h, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = static_cast<int>(bottom_blob.cstep * slice);

            const float *ptr = bottom_blob.channel(q);
            float *outptr = top_blob;
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }
        // Mat m = top_blobs[1];
        // fprintf(stdout, "#############\n");
        // for (int c = 0; c < m.c; c++)
        // {
        //     const float *ptr = m.channel(c);
        //     for (int h = 0; h < m.h; h++)
        //     {
        //         for (int w = 0; w < m.w; w++)
        //         {
        //             fprintf(stdout, "%f ", ptr[w]);
        //         }
        //         ptr += m.w;
        //         fprintf(stdout, "\n");
        //     }
        // }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((h - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, slice, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                int size = w * slice;

                float *outptr = top_blob.channel(p);
                const float *ptr = bottom_blob.channel(p).row(q);
                memcpy(outptr, ptr, size * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                float *outptr = top_blob.channel(p);
                const Mat m = bottom_blob.channel(p);

                for (int j = 0; j < h; j++)
                {
                    const float *ptr = m.row(j) + q;
                    memcpy(outptr, ptr, slice * elemsize);

                    outptr += slice;
                }
            }

            q += slice;
        }

        return 0;
    }

    return 0;
}

int Slice::forward_int8(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    const Mat &bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    const int *slices_ptr = slices;
    std::vector<int> factors;
    factors.resize(2);
    mat2vector(scales, factors);
    int right_shift = factors[1];
    int factor = factors[0];

    if (dims == 1) // axis == 0
    {
        int w = bottom_blob.w;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const signed char *ptr = (const signed char *)bottom_blob + q;
            signed char *outptr = top_blob;
            memcpy(outptr, ptr, slice * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((h - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = w * slice;

            const signed char *ptr = bottom_blob.row_signed_char(q);
            signed char *outptr = top_blob;
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < h; j++)
            {
                signed char *outptr = top_blob.row_signed_char(j);
                const signed char *ptr = bottom_blob.row_signed_char(j) + q;
                memcpy(outptr, ptr, slice * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, h, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = static_cast<int>(bottom_blob.cstep * slice);

            if (i == 0)
            {
                const signed char *ptr = bottom_blob.channel(q);
                signed char *outptr = top_blob;
                memcpy(outptr, ptr, size * elemsize);
            }
            else
            {
                for (int c = q; c < slice + q; c++)
                {
                    const signed char *ptr = bottom_blob.channel(c);

                    signed char *outptr = top_blob.channel(c - q);
                    for (int j = 0; j < bottom_blob.h * bottom_blob.w; j++)
                    {
                        if (right_shift < 0)
                        {
                            // if (name == "452")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 0.80654);
                            //     // outptr[j] = (ptr[j] * factor) >> (-right_shift);
                            // }
                            // else if (name == "484")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 10.599827 / 13.129980);
                            // }
                            // else if (name == "519")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 14.166965 / 20.343714);
                            // }
                            // else if (name == "535")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 13.292026 / 20.343714);
                            // }
                            // else if (name == "551")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 11.842315 / 20.343714);
                            // }
                            // else if (name == "567")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 14.104471 / 20.343714);
                            // }
                            // else if (name == "583")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 12.641934 / 20.343714);
                            // }
                            // else if (name == "559")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 12.998467 / 20.343714);
                            // }
                            // else if (name == "615")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 14.082044 / 20.343714);
                            // }
                            // else if (name == "650")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 19.198814 / 21.516098);
                            // }
                            // else if (name == "666")
                            // {
                            //     outptr[j] = float2int8(ptr[j] * 1.0f * 21.228539 / 21.516098);
                            // }
                            // else if (name == "468" || name == "682")
                            // {
                            //     outptr[j] = ptr[j];
                            // }
                            // else
                            {
                                outptr[j] = (ptr[j] * factor) >> (-right_shift);
                            }
                        }
                        else
                        {
                            outptr[j] = (ptr[j] * factor) << right_shift;
                        }
                    }
                }
            }
            q += slice;
        }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((h - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(w, slice, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                int size = w * slice;

                signed char *outptr = top_blob.channel(p);
                const signed char *ptr = bottom_blob.channel(p).row_signed_char(q);
                memcpy(outptr, ptr, size * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = static_cast<int>((w - q) / (top_blobs.size() - i));
            }

            Mat &top_blob = top_blobs[i];
            top_blob.create(slice, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                signed char *outptr = top_blob.channel(p);
                const Mat m = bottom_blob.channel(p);

                for (int j = 0; j < h; j++)
                {
                    const signed char *ptr = m.row_signed_char(j) + q;
                    memcpy(outptr, ptr, slice * elemsize);

                    outptr += slice;
                }
            }

            q += slice;
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
