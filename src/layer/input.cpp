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

namespace ncnn
{

DEFINE_LAYER_CREATOR(Input)

Input::Input()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = false;
}

int Input::load_param(const ParamDict &pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int Input::forward_inplace(Mat & /*bottom_top_blob*/, const Option & /*opt*/) const
{
    return 0;
}

// int Input::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
// {
//     Mat bottom_blob_int = bottom_blob;
//     if (!support_inplace)
//         return -1;

//     int w = bottom_blob.w;
//     int h = bottom_blob.h;
//     int channels = bottom_blob.c;
//     int size = w * h;
//     for (int q = 0; q < channels; q++)
//     {
//         const float *ptr = bottom_blob.channel(q);
//         int *outptr = top_blob.channel(q);

//         for (int i = 0; i < size; i++)
//         {
//             outptr[i] = int(ptr[i] * pow(2, position_scale_in));
//         }
//     }
//     top_blob = bottom_blob.clone(opt.blob_allocator);
//     if (top_blob.empty())
//         return -100;

//     return forward_inplace(top_blob, opt);
// }

} // namespace ncnn
