// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <cstdio>
#include <cstring>
#include <vector>
#include <set>
#include <map>
#include "math.h"

// ncnn public header
#include "net.h"
#include "layer.h"
#include "layer_type.h"

// ncnn private header
#include "layer/batchnorm.h"
#include "layer/bias.h"
#include "layer/binaryop.h"
#include "layer/clip.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/crop.h"
#include "layer/deconvolution.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/detectionoutput.h"
#include "layer/dropout.h"
#include "layer/eltwise.h"
#include "layer/elu.h"
#include "layer/exp.h"
#include "layer/flatten.h"
#include "layer/innerproduct.h"
#include "layer/input.h"
#include "layer/instancenorm.h"
#include "layer/interp.h"
#include "layer/log.h"
#include "layer/lrn.h"
#include "layer/lstm.h"
#include "layer/memorydata.h"
#include "layer/mvn.h"
#include "layer/normalize.h"
#include "layer/padding.h"
#include "layer/permute.h"
#include "layer/pixelshuffle.h"
#include "layer/pooling.h"
#include "layer/power.h"
#include "layer/prelu.h"
#include "layer/priorbox.h"
#include "layer/proposal.h"
#include "layer/psroipooling.h"
#include "layer/quantize.h"
#include "layer/reduction.h"
#include "layer/relu.h"
#include "layer/reorg.h"
#include "layer/requantize.h"
#include "layer/reshape.h"
#include "layer/roialign.h"
#include "layer/roipooling.h"
#include "layer/scale.h"
#include "layer/slice.h"
#include "layer/shufflechannel.h"
#include "layer/softmax.h"
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"

static bool read_int8scale_table(const char *filepath, std::map<std::string, std::vector<float>> &blob_int8scale_table, std::map<std::string, std::vector<float>> &weight_int8scale_table)
{
    blob_int8scale_table.clear();
    weight_int8scale_table.clear();

    FILE *fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "Open %s failed.\n", filepath);
        return false;
    }

    std::string key_str;
    std::vector<float> scales;

    std::vector<char> line(102400);
    char *pch = NULL;
    size_t len = 0;

    while (NULL != std::fgets(line.data(), static_cast<int>(line.size()), fp))
    {
        float scale = 1.f;
        char key[256];
        line[strcspn(line.data(), "\r\n")] = 0;

        pch = strtok(line.data(), " ");

        if (pch == NULL)
            break;

        bool is_key = true;
        while (pch != NULL)
        {
            if (is_key)
            {
                sscanf(pch, "%255s", key);

                key_str = key;
                is_key = false;
            }
            else
            {
                sscanf(pch, "%f", &scale);

                scales.push_back(scale);
            }

            pch = strtok(NULL, " ");
        }

        // XYZ_param_N pattern
        if (strstr(key_str.c_str(), "_param_"))
        {
            weight_int8scale_table[key_str] = scales;
        }
        else
        {
            blob_int8scale_table[key_str] = scales;
        }
        key_str.clear();
        scales.clear();
    }

    fclose(fp);

    return true;
}

static inline int vector_float2int(int *dst, float *src, int len, int BitN)
{
    float src_max = (src[0] >= 0) ? src[0] : (0 - src[0]);
    int i = 0;
    int Bint, Bfrac;
    float accuracy = 1.0;
    for (i = 1; i < len; i++)
    {
        float tmp = (src[i] >= 0) ? src[i] : (0 - src[i]);
        if (src_max < tmp)
        {
            src_max = tmp;
        }
    }

    Bint = (int)(log2(src_max));
    Bfrac = BitN - Bint;
    accuracy = pow((float)2.0, (float)(0 - Bfrac));
    for (i = 0; i < len; i++)
    {
        dst[i] = int(src[i] / accuracy + 0.5);
    }
    return (0 - Bfrac);
}

class NetQuantize : public ncnn::Net
{
public:
    // 0=fp32 1=fp16 2=int8
    int storage_type;
    std::map<std::string, std::vector<float>> blob_int8scale_table;
    std::map<std::string, std::vector<float>> weight_int8scale_table;
    std::map<std::string, std::vector<signed char>> layers_scale;
    std::map<std::string, std::vector<float>> top_blob_int8scale_table;
    std::map<std::string, std::vector<float>> result_blob_int8scale_table;
    std::map<std::string, std::vector<int>> int_result_blob_int8scale_table;
    // std::map<std::string, int> top_scale_right_shift;
    std::map<std::string, std::vector<int>> int_top_blob_int8scale_table;
    std::map<std::string, std::vector<int>> split_factor_table;

public:
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();
    int find_top_scale();
    std::string find_first_conv(std::string name);
    std::string find_first_shufflechannel(std::string name);
    void fill_scale_shufflechannel(std::vector<float> &layer_scale, int expected_layer_size, std::vector<int> status, std::set<int> &visited, std::string layer_name, bool left);
    void split_scale(std::string name, std::vector<float> top_scales);
    std::string find_first_split(std::string name);

public:
    int fprintf_param_int_array(int id, const ncnn::Mat &m, FILE *pp);
    int fprintf_param_float_array(int id, const ncnn::Mat &m, FILE *pp);

    int fwrite_weight_tag_data(int tag, const ncnn::Mat &data, FILE *bp);
    int fwrite_weight_data(const ncnn::Mat &data, FILE *bp);

    int save(const char *parampath, const char *binpath);
};

int NetQuantize::find_top_scale()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        ncnn::Layer *layer = layers[i];
        if (layer->type == "Input")
        {
            std::string layer_name = layer->name;
            for (size_t n = 0; n < blobs[layer->tops[0]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[0]].consumers[n];
                ncnn::Layer *layer_next = layers[layer_next_index];
                if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise")
                {
                    top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[layer_next->name][0]);
                }
            }
        }

        if (i == layer_count - 1 && layer->type == "InnerProduct")
        {
            std::string layer_name = layer->name;
            top_blob_int8scale_table[layer_name].push_back(1.0);
        }

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            std::string layer_name = layer->name;
            //char weight_key[256];
            //sprintf(weight_key, "%s_param_0", layer_name.c_str());
            // std::map<std::string, std::vector<float>>::iterator iter_data = blob_int8scale_table.find(layer_name);
            // std::map<std::string, std::vector<float>>::iterator iter = weight_int8scale_table.find(weight_key);
            // std::vector<float> weight_data_int8_scales = iter->second;
            // std::vector<float> blob_int8_scales = iter_data->second;
            // fprintf(stderr, "forward_layer %s\n", layer->name.c_str());
            for (size_t n = 0; n < blobs[layer->tops[0]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[0]].consumers[n];
                ncnn::Layer *layer_next = layers[layer_next_index];
                if (layer_next->type == "Split")
                {
                    bool all_conv = true;
                    std::string layer_next_2_name = "NULL";
                    for (size_t i = 0; i < layer_next->tops.size(); i++)
                    {
                        int layer_next_2_index = blobs[layer_next->tops[i]].consumers[0];
                        if (layers[layer_next_2_index]->type != "Convolution" && layers[layer_next_2_index]->type != "ConvolutionDepthWise")
                        {
                            // fprintf(stderr, "%s, %s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str(), layers[layer_next_3_index]->name.c_str());
                            all_conv = false;
                        }
                        if (all_conv)
                        {
                            layer_next_2_name = layers[layer_next_2_index]->name;
                        }
                    }
                    bool condition1 = all_conv;

                    bool condition2 = false;
                    std::string layer_next_2_name_0 = "NULL"; // Conv/ConvDW
                    std::string layer_next_2_name_1 = "NULL"; // Binary
                    {
                        if (layer_next->tops.size() == 2)
                        {
                            if ((layers[blobs[layer_next->tops[0]].consumers[0]]->type == "Convolution" ||
                                 layers[blobs[layer_next->tops[0]].consumers[0]]->type == "ConvolutionDepthWise") &&
                                (layers[blobs[layer_next->tops[1]].consumers[0]]->type == "BinaryOp" || layers[blobs[layer_next->tops[1]].consumers[0]]->type == "Padding"))
                            {
                                layer_next_2_name_0 = layers[blobs[layer_next->tops[0]].consumers[0]]->name;
                                layer_next_2_name_1 = layers[blobs[layer_next->tops[1]].consumers[0]]->name;
                                condition2 = true;
                            }
                            else if ((layers[blobs[layer_next->tops[1]].consumers[0]]->type == "Convolution" ||
                                      layers[blobs[layer_next->tops[1]].consumers[0]]->type == "ConvolutionDepthWise") &&
                                     (layers[blobs[layer_next->tops[0]].consumers[0]]->type == "BinaryOp" || layers[blobs[layer_next->tops[0]].consumers[0]]->type == "Padding"))
                            {
                                layer_next_2_name_0 = layers[blobs[layer_next->tops[1]].consumers[0]]->name;
                                layer_next_2_name_1 = layers[blobs[layer_next->tops[0]].consumers[0]]->name;
                                condition2 = true;
                            }
                        }
                    }

                    bool condition3 = false;

                    // condition 1
                    if (condition1)
                    {
                        top_blob_int8scale_table[layer_name] = blob_int8scale_table[layer_next_2_name];
                    }
                    // condition 2
                    else if (condition2)
                    {
                        // if (layer->name == "379" || layer->name == "383") //TODO
                        // {
                        //     std::string first_conv_name = find_first_conv(layer_next_2_name_1);
                        //     top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[layer_next_2_name_0][0]);
                        //     top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                        // }
                        // else
                        // {
                        top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[layer_next_2_name_0][0]);
                        std::string first_conv_name = find_first_conv(layer_next_2_name_1);
                        top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                        // }
                    }
                    else
                    {
                        fprintf(stderr, "Condition have not be considered\n");
                        return -1;
                    }
                }
                else if (layer_next->type == "Concat")
                {
                    bool condition3 = false;
                    for (size_t i = 0; i < layer_next->tops.size(); i++)
                    {
                        int layer_next_2_index = blobs[layer_next->tops[i]].consumers[0];
                        ncnn::Layer *layer_next2 = layers[layer_next_2_index];
                        if (layer_next2->type == "ShuffleChannel")
                        {
                            for (size_t i = 0; i < layer_next2->tops.size(); i++)
                            {
                                int layer_next_3_index = blobs[layer_next2->tops[i]].consumers[0];
                                if (layers[layer_next_3_index]->type == "Slice")
                                {
                                    condition3 = true;
                                }
                            }
                        }
                    }
                    if (condition3)
                    {
                        int out_channels = 0;
                        if (layer->type == "Convolution")
                        {
                            out_channels = ((ncnn::Convolution *)layer)->num_output;
                        }
                        else
                        {
                            out_channels = ((ncnn::ConvolutionDepthWise *)layer)->num_output;
                        }
                        std::vector<float> layer_scale;
                        layer_scale.resize(out_channels * 2);
                        std::vector<int> status;
                        for (int i = 0; i < out_channels * 2; i++)
                        {
                            status.push_back(i);
                        }
                        std::set<int> visited;
                        // conv on left or right
                        // if (layer_name == "673")
                        //     fprintf(stdout, "673\n");
                        bool left = true;
                        {
                            if (layer_next->bottoms.size() == 2)
                            {
                                // int debug_index = find_layer_index_by_name(layer->name.c_str());
                                // int debug_index2 = blobs[layer_next->bottoms[1]].producer;
                                if (find_layer_index_by_name(layer->name.c_str()) == blobs[layer_next->bottoms[1]].producer)
                                {
                                    left = false;
                                }
                            }
                        }
                        fill_scale_shufflechannel(layer_scale, out_channels, status, visited, layer->name, left);

                        if (left)
                        {
                            for (int i = 0; i < layer_scale.size() / 2; i++)
                            {
                                top_blob_int8scale_table[layer_name].push_back(layer_scale[i]);
                            }
                        }
                        else
                        {
                            for (int i = layer_scale.size() / 2 - 1; i < layer_scale.size(); i++)
                            {
                                top_blob_int8scale_table[layer_name].push_back(layer_scale[i]);
                            }
                        }
                        continue;
                    }
                    else
                    {
                        std::string first_conv_name = find_first_conv(layer->name);
                        if (blob_int8scale_table.count(first_conv_name) > 0)
                        {
                            top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                        }
                        else
                        {
                            // if not find conv is rest layer, it will be fake number 1.0
                            top_blob_int8scale_table[layer_name].push_back(1.0);
                        }
                    }
                }
                else if (layer_next->type == "BinaryOp")
                {
                    int layer_next_1_index = blobs[layer_next->tops[0]].consumers[n];
                    ncnn::Layer *layer_next_1 = layers[layer_next_1_index];
                    if (layer_next_1->type == "Split")
                    {
                        bool all_conv = true;
                        std::string layer_next_2_name = "NULL";
                        for (size_t i = 0; i < layer_next_1->tops.size(); i++)
                        {
                            int layer_next_2_index = blobs[layer_next_1->tops[i]].consumers[0];
                            if (layers[layer_next_2_index]->type != "Convolution" && layers[layer_next_2_index]->type != "ConvolutionDepthWise")
                            {
                                // fprintf(stderr, "%s, %s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str(), layers[layer_next_3_index]->name.c_str());
                                all_conv = false;
                            }
                            if (all_conv)
                            {
                                layer_next_2_name = layers[layer_next_2_index]->name;
                            }
                        }
                        bool condition1 = all_conv;

                        bool condition2 = false;
                        std::string layer_next_2_name_0 = "NULL"; // Conv/ConvDW
                        std::string layer_next_2_name_1 = "NULL"; // Binary
                        {
                            if (layer_next_1->tops.size() == 2)
                            {
                                if ((layers[blobs[layer_next_1->tops[0]].consumers[0]]->type == "Convolution" ||
                                     layers[blobs[layer_next_1->tops[0]].consumers[0]]->type == "ConvolutionDepthWise") &&
                                    (layers[blobs[layer_next_1->tops[1]].consumers[0]]->type == "BinaryOp" || layers[blobs[layer_next_1->tops[1]].consumers[0]]->type == "Padding"))
                                {
                                    layer_next_2_name_0 = layers[blobs[layer_next_1->tops[0]].consumers[0]]->name;
                                    layer_next_2_name_1 = layers[blobs[layer_next_1->tops[1]].consumers[0]]->name;
                                    condition2 = true;
                                }
                                else if ((layers[blobs[layer_next_1->tops[1]].consumers[0]]->type == "Convolution" ||
                                          layers[blobs[layer_next_1->tops[1]].consumers[0]]->type == "ConvolutionDepthWise") &&
                                         (layers[blobs[layer_next_1->tops[0]].consumers[0]]->type == "BinaryOp" || layers[blobs[layer_next_1->tops[0]].consumers[0]]->type == "Padding"))
                                {
                                    layer_next_2_name_0 = layers[blobs[layer_next_1->tops[1]].consumers[0]]->name;
                                    layer_next_2_name_1 = layers[blobs[layer_next_1->tops[0]].consumers[0]]->name;
                                    condition2 = true;
                                }
                            }
                        }

                        bool condition3 = false;

                        // condition 1
                        if (condition1)
                        {
                            top_blob_int8scale_table[layer_name] = blob_int8scale_table[layer_next_2_name];
                        }
                        // condition 2
                        else if (condition2)
                        {
                            top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[layer_next_2_name_0][0]);
                            std::string first_conv_name = find_first_conv(layer_next_2_name_1);
                            top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                        }
                        else
                        {
                            fprintf(stderr, "Condition have not be considered\n");
                            return -1;
                        }
                    }
                    else
                    {
                        std::string first_conv_name = find_first_conv(layer->name);
                        if (blob_int8scale_table.count(first_conv_name) > 0)
                        {
                            top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                        }
                        else
                        {
                            // if not find conv is rest layer, it will be fake number 1.0
                            top_blob_int8scale_table[layer_name].push_back(1.0);
                        }
                    }
                }
                else
                {
                    // for last condition 4, convolution have no output scale, fake number 1.0
                    std::string first_conv_name = find_first_conv(layer->name);
                    if (blob_int8scale_table.count(first_conv_name) > 0)
                    {
                        top_blob_int8scale_table[layer_name].push_back(blob_int8scale_table[first_conv_name][0]);
                    }
                    else
                    {
                        // if not find conv is rest layer, it will be fake number 1.0
                        top_blob_int8scale_table[layer_name].push_back(1.0);
                    }
                }
            }
        }
    }
    return 0;
}
std::string NetQuantize::find_first_conv(std::string name)
{
    ncnn::Layer *layer = layers[find_layer_index_by_name(name.c_str())];
    // if this layer is last layer
    if (blobs[layer->tops[0]].consumers.size() == 0)
    {
        return layer->name;
    }

    if (layer->type == "Slice")
    {
        int layer_next_index = blobs[layer->tops[1]].consumers[0];
        ncnn::Layer *layer_next = layers[layer_next_index];
        if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise" || layer_next->type == "InnerProduct")
        {
            return layer_next->name;
        }
        else
        {
            return find_first_conv(layer_next->name);
        }
    }
    else
    {
        if (layer->type == "Split")
        {
            for (size_t n = 0; n < blobs[layer->tops[1]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[1]].consumers[n];
                ncnn::Layer *layer_next = layers[layer_next_index];
                if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise" || layer_next->type == "InnerProduct")
                {
                    return layer_next->name;
                }
                else
                {
                    return find_first_conv(layer_next->name);
                }
            }
        }
        else
        {
            for (size_t n = 0; n < blobs[layer->tops[0]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[0]].consumers[n];
                ncnn::Layer *layer_next = layers[layer_next_index];
                if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise" || layer_next->type == "InnerProduct")
                {
                    return layer_next->name;
                }
                else
                {
                    return find_first_conv(layer_next->name);
                }
            }
        }
    }
}

std::string NetQuantize::find_first_shufflechannel(std::string name)
{
    ncnn::Layer *layer = layers[find_layer_index_by_name(name.c_str())];
    // if this layer is last layer
    if (blobs[layer->tops[0]].consumers.size() == 0)
    {
        fprintf(stderr, "Nor found shufflechannel layer\n");
        return layer->name;
    }
    for (size_t n = 0; n < blobs[layer->tops[0]].consumers.size(); n++)
    {
        int layer_next_index = blobs[layer->tops[0]].consumers[n];
        ncnn::Layer *layer_next = layers[layer_next_index];
        if (layer_next->type == "ShuffleChannel")
        {
            return layer_next->name;
        }
        else
        {
            return find_first_shufflechannel(layer_next->name);
        }
    }
}

std::string NetQuantize::find_first_split(std::string name)
{
    ncnn::Layer *layer = layers[find_layer_index_by_name(name.c_str())];
    // if this layer is last layer
    if (blobs[layer->tops[0]].consumers.size() == 0)
    {
        fprintf(stderr, "Nor found shufflechannel layer\n");
        return layer->name;
    }
    for (size_t n = 0; n < blobs[layer->tops[0]].consumers.size(); n++)
    {
        int layer_next_index = blobs[layer->tops[0]].consumers[n];
        ncnn::Layer *layer_next = layers[layer_next_index];
        if (layer_next->type == "Split")
        {
            return layer_next->name;
        }
        else
        {
            return find_first_split(layer_next->name);
        }
    }
}

void NetQuantize::fill_scale_shufflechannel(std::vector<float> &layer_scale, int expected_layer_size, std::vector<int> status, std::set<int> &visited, std::string layer_name, bool left)
{
    // fprintf(stderr, "forward_layer %s\n", layer_name.c_str());
    // for (int i = 0; i < layer_scale.size(); i++)
    // {
    //     fprintf(stdout, "%f ", layer_scale[i]);
    // }
    // fprintf(stdout, "\n");

    if (visited.size() == expected_layer_size)
        return;

    //[ 0, 1, 2, 3, 4, 5, 6, 7 ]
    // status [0 4 1 5 2 6 3 7]
    // new_status  [0 2 4 6 1 3 5 7]
    //int first_shufflechannel_index = find_layer_index_by_name(find_first_shufflechannel(layer_name));

    ncnn::Layer *layer = layers[find_layer_index_by_name(layer_name.c_str())];
    if (layer->type == "ShuffleChannel")
    {
        int size = ((ncnn::ShuffleChannel *)layer)->num_output;
        std::vector<int> new_status;
        new_status.resize(size);
        // fprintf(stdout, "mm%d-%d", size, status.size());
        int chs_per_group = size / 2;
        int layer_next_index = blobs[layer->tops[0]].consumers[0];
        ncnn::Layer *layer_next = layers[layer_next_index];

        // if use shufflechannel, need change status
        std::vector<int> now;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < chs_per_group; j++)
            {
                int src = chs_per_group * i + j;
                int dst = 2 * j + i;
                new_status[dst] = status[src];
            }
        }

        if (layer_next->type == "Slice")
        {
            for (int i = size - 1; i >= size / 2; i--)
            {
                if (left && new_status[i] < chs_per_group)
                {
                    now.push_back(new_status[i]);
                }

                if (!left && new_status[i] >= chs_per_group)
                {
                    now.push_back(new_status[i]);
                }
            }

            for (int i = 1; i >= 0; i--)
            {
                if (i == 1)
                {
                    for (int j = 0; j < now.size(); j++)
                    {
                        if (!visited.count(now[j]))
                        {

                            // get scale
                            std::string conv_name = find_first_conv(layer_name);

                            layer_scale[now[j]] = blob_int8scale_table.find(conv_name)->second[0];
                            visited.insert(now[j]);
                        }
                    }
                }
                else
                {
                    for (size_t k = 0; k < layer->tops.size(); k++)
                    {
                        int layer_next_index = blobs[layer->tops[k]].consumers[0];
                        ncnn::Layer *layer_next = layers[layer_next_index];
                        fill_scale_shufflechannel(layer_scale, expected_layer_size, new_status, visited, layer_next->name, left);
                        return;
                    }
                }
            }

            /*
            for (int i = 1; i >= 0; i--)
            {
                for (int j = 0; j < chs_per_group; j++)
                {
                    int src = chs_per_group * i + j;
                    int dst = 2 * j + i;
                    //new_status.push_back(status[dst]);
                    if (i == 1)
                    {
                        if (left && (src < chs_per_group))
                        {
                            if (!visited.count(status[dst]))
                            {
                                visited.insert(status[dst]);
                                // get scale
                                std::string conv_name = find_first_conv(layer_name);

                                layer_scale[status[dst]] = blob_int8scale_table.find(conv_name)->second[0];
                            }
                        }

                        if (!left && (src >= chs_per_group))
                        {
                            if (!visited.count(status[dst]))
                            {
                                visited.insert(status[dst]);
                                // get scale
                                std::string conv_name = find_first_conv(layer_name);
                                layer_scale[status[dst]] = blob_int8scale_table.find(conv_name)->second[0];
                            }
                        }
                    }
                    else
                    {
                        for (size_t k = 0; k < layer->tops.size(); k++)
                        {
                            int layer_next_index = blobs[layer->tops[k]].consumers[0];
                            ncnn::Layer *layer_next = layers[layer_next_index];
                            fill_scale_shufflechannel(layer_scale, expected_layer_size, status, visited, layer_next->name, left);
                        }
                        //return;
                        // if s(!visited.count(status[dst]))
                        // {
                        //     visited.insert(status[dst]);
                        // }
                    }
                }
            }
            */
            // for (int i = 0; i < 2; i++)
            // {
            //     for (int j = 0; j < chs_per_group; j++)
            //     {
            //         if (i == 0)
            //         {
            //             for (size_t i = 0; i < layer->tops.size(); i++)
            //             {
            //                 int layer_next_index = blobs[layer->tops[i]].consumers[0];
            //                 ncnn::Layer *layer_next = layers[layer_next_index];
            //                 fill_scale_shufflechannel(layer_scale, expected_layer_size, status, visited, layer_next->name, left);
            //             }
            //             return;
            //         }
            //     }
            // }
            //status.swap(new_status);
            return;
        }
        else if (layer_next->type == "Split")
        {

            for (int j = 0; j < status.size(); j++)
            {
                if (!visited.count(status[j]))
                {

                    // get scale
                    std::string conv_name = find_first_conv(layer_name);

                    layer_scale[status[j]] = blob_int8scale_table.find(conv_name)->second[0];
                    visited.insert(status[j]);
                }
            }
            return;
        }
        else if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise" || layer_next->type == "InnerProduct")
        {
            for (int i = 0; i < layer_scale.size(); i++)
            {
                if (!visited.count(i))
                    layer_scale[i] = blob_int8scale_table[layer_next->name][0];
            }
            return;
        }
        else
        {
            fprintf(stderr, "Not consider this condition %s\n", layer_name.c_str());
            return;
        }
    }
    else
    {
        for (size_t i = layer->tops.size() - 1; i >= 0; i--)
        {
            int layer_next_index = blobs[layer->tops[i]].consumers[0];
            ncnn::Layer *layer_next = layers[layer_next_index];
            fill_scale_shufflechannel(layer_scale, expected_layer_size, status, visited, layer_next->name, left);
            return;
        }
        return;
    }
}

void NetQuantize::split_scale(std::string name, std::vector<float> top_scales)
{
    float left_scale = top_scales[0];
    float right_scale = top_scales[1];
    int left = 1;
    float factor;
    int Bint;
    int BitN = 7;
    int Bfrac;
    int right_shift;
    int dst;
    std::vector<int> factors;
    factors.resize(3);
    std::string split_name;
    float accuracy;
    if (left_scale < right_scale)
    {
        left = -1;
        factor = left_scale / right_scale;
    }
    else
    {
        factor = right_scale / left_scale;
    }

    //Bint = (int)(log2(factor) + 0.5);
    Bint = (int)(log2(factor) + 0.5);
    Bfrac = BitN - Bint;
    accuracy = pow((float)2.0, (float)(0 - Bfrac));
    dst = int(factor / accuracy + 0.5);
    right_shift = 0 - Bfrac;
    factors[0] = left;
    factors[1] = dst;
    factors[2] = right_shift;
    fprintf(stdout, "xxxx %f %d-%d $ %d\n", factor, left, dst, right_shift);

    split_name = find_first_split(name);
    split_factor_table[split_name] = factors;
}

int NetQuantize::quantize_convolution()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "Convolution")
            continue;

        // find convolution layer
        std::map<std::string, std::vector<float>>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float>>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::Convolution *convolution = (ncnn::Convolution *)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;
        std::vector<float> blob_int8_scales = iter_data->second;

        fprintf(stderr, "quantize_convolution %s\n", convolution->name.c_str());

        {
            ncnn::Mat int8_weight_data(convolution->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convolution->weight_data_size / convolution->num_output;

            // quantize weight to int8
            for (int n = 0; n < convolution->num_output; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convolution->weight_data = int8_weight_data;
        }

        // quantize bias to int32
        if (convolution->bias_term)
        {
            ncnn::Mat int32_bias_data(convolution->num_output, (size_t)4u); // 4u -> int32
            if (int32_bias_data.empty())
                return -100;

            for (int n = 0; n < convolution->num_output; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n] * blob_int8_scales[0]); // bias_scale = weight_scale * bottom_blob_scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int32_bias_data.allocator;
                opt.use_int32_storage = true;

                const ncnn::Mat bias_data_n = convolution->bias_data.range(n, 1);
                ncnn::Mat int32_bias_data_n = int32_bias_data.range(n, 1);
                op->forward(bias_data_n, int32_bias_data_n, opt);

                delete op;
            }

            // ncnn::Mat m = int32_bias_data;
            // for (int c = 0; c < m.c; c++)
            // {
            //     const int *ptr = m.channel(c);
            //     for (int h = 0; h < m.h; h++)
            //     {
            //         for (int w = 0; w < m.w; w++)
            //         {
            //             fprintf(stdout, "%d ", ptr[w]);
            //         }
            //         ptr += m.w;
            //         fprintf(stdout, "\n");
            //     }
            // }
            convolution->bias_data = int32_bias_data;
        }

        convolution->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::quantize_convolutiondepthwise()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // find convolutiondepthwise layer
        std::map<std::string, std::vector<float>>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float>>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::ConvolutionDepthWise *convdw = (ncnn::ConvolutionDepthWise *)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;
        std::vector<float> blob_int8_scales = iter_data->second;

        fprintf(stderr, "quantize_convolution %s\n", convdw->name.c_str());

        {
            ncnn::Mat int8_weight_data(convdw->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convdw->weight_data_size / convdw->group;

            // quantize weight to int8
            for (int n = 0; n < convdw->group; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = convdw->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convdw->weight_data = int8_weight_data;
        }

        // quantize bias to int8 will cause big error, quantize bias to int32
        if (convdw->bias_term)
        {
            ncnn::Mat int32_bias_data(convdw->num_output, (size_t)4u); // 4u -> int32
            if (int32_bias_data.empty())
                return -100;

            for (int n = 0; n < convdw->num_output; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n] * blob_int8_scales[0]); // bias_scale = weight_scale * bottom_blob_scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int32_bias_data.allocator;
                opt.use_int32_storage = true;

                const ncnn::Mat bias_data_n = convdw->bias_data.range(n, 1);
                ncnn::Mat int32_bias_data_n = int32_bias_data.range(n, 1);
                op->forward(bias_data_n, int32_bias_data_n, opt);

                delete op;
            }

            convdw->bias_data = int32_bias_data;
        }

        convdw->int8_scale_term = 1;
    }

    return 0;
}

int NetQuantize::quantize_innerproduct()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "InnerProduct")
            continue;

        // find InnerProduct layer
        std::map<std::string, std::vector<float>>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float>>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // InnerProduct - quantize weight from fp32 to int8
        ncnn::InnerProduct *fc = (ncnn::InnerProduct *)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;
        std::vector<float> blob_int8_scales = iter_data->second;

        fprintf(stderr, "quantize_convolution %s\n", fc->name.c_str());

        {
            ncnn::Mat int8_weight_data(fc->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = fc->weight_data_size / fc->num_output;

            // quantize weight to int8
            for (int n = 0; n < fc->num_output; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = fc->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            fc->weight_data = int8_weight_data;
        }

        // quantize bias to int32
        if (fc->bias_term)
        {
            ncnn::Mat int32_bias_data(fc->num_output, (size_t)4u); // 4u -> int32
            if (int32_bias_data.empty())
                return -100;

            for (int n = 0; n < fc->num_output; n++)
            {
                ncnn::Layer *op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n] * blob_int8_scales[0]); // bias_scale = weight_scale * bottom_blob_scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int32_bias_data.allocator;
                opt.use_int32_storage = true;

                const ncnn::Mat bias_data_n = fc->bias_data.range(n, 1);
                ncnn::Mat int32_bias_data_n = int32_bias_data.range(n, 1);
                op->forward(bias_data_n, int32_bias_data_n, opt);

                delete op;
            }

            fc->bias_data = int32_bias_data;
        }

        fc->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::fprintf_param_int_array(int id, const ncnn::Mat &m, FILE *pp)
{
    const int count = m.w;
    const int *ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%d", ptr[i]);
    }

    return 0;
}

int NetQuantize::fprintf_param_float_array(int id, const ncnn::Mat &m, FILE *pp)
{
    const int count = m.w;
    const float *ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%f", ptr[i]);
    }

    return 0;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

int NetQuantize::fwrite_weight_tag_data(int tag, const ncnn::Mat &data, FILE *bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);

    if (data.elemsize == 1)
        tag = 0x000D4B38; // int8 magic

    fwrite(&tag, sizeof(int), 1, bp);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = static_cast<int>(alignSize(nwrite, 4));
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::fwrite_weight_data(const ncnn::Mat &data, FILE *bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = static_cast<int>(alignSize(nwrite, 4));
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::save(const char *parampath, const char *binpath)
{
    FILE *pp = fopen(parampath, "wb");
    FILE *bp = fopen(binpath, "wb");

    fprintf(pp, "7767517\n");

    const int layer_count = static_cast<int>(layers.size());

    int layer_count_fused = 0;
    std::set<std::string> blob_names;
    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer *layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer_count_fused++;

        int bottom_count = static_cast<int>(layer->bottoms.size());
        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            blob_names.insert(blobs[bottom_blob_index].name);
        }

        int top_count = static_cast<int>(layer->tops.size());
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            blob_names.insert(blobs[top_blob_index].name);
        }
    }

    int blob_count_fused = static_cast<int>(blob_names.size());

    fprintf(pp, "%d %d\n", layer_count_fused, blob_count_fused);

    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer *layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        int bottom_count = static_cast<int>(layer->bottoms.size());
        int top_count = static_cast<int>(layer->tops.size());

        fprintf(pp, "%-24s %-24s %d %d", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            fprintf(pp, " %s", blobs[bottom_blob_index].name.c_str());
        }
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            fprintf(pp, " %s", blobs[top_blob_index].name.c_str());
        }

        ncnn::Layer *layer_default = ncnn::create_layer(layer->typeindex);

        ncnn::ParamDict pd;
        layer_default->load_param(pd);

#define fprintf_param_value(format, phase)  \
    {                                       \
        if (op->phase != op_default->phase) \
            fprintf(pp, format, op->phase); \
    }

        if (layer->type == "BatchNorm")
        {
            ncnn::BatchNorm *op = (ncnn::BatchNorm *)layer;
            ncnn::BatchNorm *op_default = (ncnn::BatchNorm *)layer_default;

            fprintf_param_value(" 0=%d", channels)
                fprintf_param_value(" 1=%f", eps)

                    fwrite_weight_data(op->slope_data, bp);
            fwrite_weight_data(op->mean_data, bp);
            fwrite_weight_data(op->var_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "Bias")
        {
            ncnn::Bias *op = (ncnn::Bias *)layer;
            ncnn::Bias *op_default = (ncnn::Bias *)layer_default;

            fprintf_param_value(" 0=%d", bias_data_size)

                fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "BinaryOp")
        {
            ncnn::BinaryOp *op = (ncnn::BinaryOp *)layer;
            ncnn::BinaryOp *op_default = (ncnn::BinaryOp *)layer_default;

            fprintf_param_value(" 0=%d", op_type)
                fprintf_param_value(" 1=%d", with_scalar)
                    fprintf_param_value(" 2=%f", b)
                        fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Clip")
        {
            ncnn::Clip *op = (ncnn::Clip *)layer;
            ncnn::Clip *op_default = (ncnn::Clip *)layer_default;

            fprintf_param_value(" 0=%f", min)
                fprintf_param_value(" 1=%f", max)
                    fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Concat")
        {
            ncnn::Concat *op = (ncnn::Concat *)layer;
            ncnn::Concat *op_default = (ncnn::Concat *)layer_default;

            fprintf_param_value(" 0=%d", axis)
                fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Convolution")
        {
            ncnn::Convolution *op = (ncnn::Convolution *)layer;
            ncnn::Convolution *op_default = (ncnn::Convolution *)layer_default;

            fprintf_param_value(" 0=%d", num_output)
                fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w)
                    fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w)
                    fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w)
                    fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left)
                    fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left)
                    fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top)
                    fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 5=%d", bias_term)
                fprintf_param_value(" 6=%d", weight_data_size)
                    fprintf_param_value(" 8=%d", int8_scale_term)
                        fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty())
                    fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {
                // std::vector<float> weight_int8scale;
                // std::vector<float> blob_int8scale;

                // char key[256];
                // sprintf(key, "%s_param_0", layers[i]->name.c_str());

                // if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                // {
                //     weight_int8scale = weight_int8scale_table[std::string(key)];
                // }

                // if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                // {
                //     blob_int8scale = blob_int8scale_table[layer->name];
                // }

                // // write int8_scale data
                // fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                // fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);

                std::vector<int> int_scales;
                int right_shift;
                if (int_result_blob_int8scale_table.find(layer->name) != int_result_blob_int8scale_table.end())
                {
                    int_scales = int_result_blob_int8scale_table[layer->name];
                }

                // if (top_scale_right_shift.find(layer->name) != top_scale_right_shift.end())
                // {
                //     right_shift = top_scale_right_shift[layer->name];
                // }
                std::vector<int> int_top_scales;
                if (int_top_blob_int8scale_table.find(layer->name) != int_top_blob_int8scale_table.end())
                {
                    int_top_scales = int_top_blob_int8scale_table[layer->name];
                }

                // write int8_scale data and shift
                fwrite(int_scales.data(), sizeof(int), int_scales.size(), bp);
                fwrite(int_top_scales.data(), sizeof(int), int_top_scales.size(), bp);
                fprintf(pp, " 18=%d", int_top_scales.size());
                // fwrite(&right_shift, sizeof(int), 1, bp);
            }
        }
        else if (layer->type == "ConvolutionDepthWise")
        {
            ncnn::ConvolutionDepthWise *op = (ncnn::ConvolutionDepthWise *)layer;
            ncnn::ConvolutionDepthWise *op_default = (ncnn::ConvolutionDepthWise *)layer_default;

            fprintf_param_value(" 0=%d", num_output)
                fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w)
                    fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w)
                    fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w)
                    fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left)
                    fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left)
                    fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top)
                    fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 5=%d", bias_term)
                fprintf_param_value(" 6=%d", weight_data_size)
                    fprintf_param_value(" 7=%d", group)
                        fprintf_param_value(" 8=%d", int8_scale_term)
                            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty())
                    fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {
                // std::vector<float> weight_int8scale;
                // std::vector<float> blob_int8scale;

                // char key[256];
                // sprintf(key, "%s_param_0", layers[i]->name.c_str());

                // if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                // {
                //     weight_int8scale = weight_int8scale_table[std::string(key)];
                // }

                // if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                // {
                //     blob_int8scale = blob_int8scale_table[layer->name];
                // }

                // // write int8_scale data
                // fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                // fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);

                std::vector<int> int_scales;
                int right_shift;
                if (int_result_blob_int8scale_table.find(layer->name) != int_result_blob_int8scale_table.end())
                {
                    int_scales = int_result_blob_int8scale_table[layer->name];
                }
                // if (top_scale_right_shift.find(layer->name) != top_scale_right_shift.end())
                // {
                //     right_shift = top_scale_right_shift[layer->name];
                // }

                std::vector<int> int_top_scales;
                if (int_top_blob_int8scale_table.find(layer->name) != int_top_blob_int8scale_table.end())
                {
                    int_top_scales = int_top_blob_int8scale_table[layer->name];
                }
                // write int8_scale data and shift
                fwrite(int_scales.data(), sizeof(int), int_scales.size(), bp);
                fwrite(int_top_scales.data(), sizeof(int), int_top_scales.size(), bp);
                fprintf(pp, " 18=%d", int_top_scales.size());
                // fwrite(&right_shift, sizeof(int), 1, bp);
            }
        }
        else if (layer->type == "Crop")
        {
            ncnn::Crop *op = (ncnn::Crop *)layer;
            ncnn::Crop *op_default = (ncnn::Crop *)layer_default;

            fprintf_param_value(" 0=%d", woffset)
                fprintf_param_value(" 1=%d", hoffset)
                    fprintf_param_value(" 2=%d", coffset)
                        fprintf_param_value(" 3=%d", outw)
                            fprintf_param_value(" 4=%d", outh)
                                fprintf_param_value(" 5=%d", outc)
                                    fprintf_param_value(" 6=%d", woffset2)
                                        fprintf_param_value(" 7=%d", hoffset2)
                                            fprintf_param_value(" 8=%d", coffset2)
            {
                if (!op->starts.empty())
                    fprintf_param_int_array(9, op->starts, pp);
            }
            {
                if (!op->ends.empty())
                    fprintf_param_int_array(10, op->ends, pp);
            }
            {
                if (!op->axes.empty())
                    fprintf_param_int_array(11, op->axes, pp);
            }
        }
        else if (layer->type == "Deconvolution")
        {
            ncnn::Deconvolution *op = (ncnn::Deconvolution *)layer;
            ncnn::Deconvolution *op_default = (ncnn::Deconvolution *)layer_default;

            fprintf_param_value(" 0=%d", num_output)
                fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w)
                    fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w)
                    fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w)
                    fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left)
                    fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left)
                    fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top)
                    fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 5=%d", bias_term)
                fprintf_param_value(" 6=%d", weight_data_size)
                    fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty())
                    fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "DeconvolutionDepthWise")
        {
            ncnn::DeconvolutionDepthWise *op = (ncnn::DeconvolutionDepthWise *)layer;
            ncnn::DeconvolutionDepthWise *op_default = (ncnn::DeconvolutionDepthWise *)layer_default;

            fprintf_param_value(" 0=%d", num_output)
                fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w)
                    fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w)
                    fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w)
                    fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left)
                    fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left)
                    fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top)
                    fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 5=%d", bias_term)
                fprintf_param_value(" 6=%d", weight_data_size)
                    fprintf_param_value(" 7=%d", group)
                        fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty())
                    fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "DetectionOutput")
        {
            ncnn::DetectionOutput *op = (ncnn::DetectionOutput *)layer;
            ncnn::DetectionOutput *op_default = (ncnn::DetectionOutput *)layer_default;

            fprintf_param_value(" 0=%d", num_class)
                fprintf_param_value(" 1=%f", nms_threshold)
                    fprintf_param_value(" 2=%d", nms_top_k)
                        fprintf_param_value(" 3=%d", keep_top_k)
                            fprintf_param_value(" 4=%f", confidence_threshold)
                                fprintf_param_value(" 5=%f", variances[0])
                                    fprintf_param_value(" 6=%f", variances[1])
                                        fprintf_param_value(" 7=%f", variances[2])
                                            fprintf_param_value(" 8=%f", variances[3])
        }
        else if (layer->type == "Dropout")
        {
            ncnn::Dropout *op = (ncnn::Dropout *)layer;
            ncnn::Dropout *op_default = (ncnn::Dropout *)layer_default;

            fprintf_param_value(" 0=%f", scale)
        }
        else if (layer->type == "Eltwise")
        {
            ncnn::Eltwise *op = (ncnn::Eltwise *)layer;
            ncnn::Eltwise *op_default = (ncnn::Eltwise *)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            {
                if (!op->coeffs.empty())
                    fprintf_param_float_array(1, op->coeffs, pp);
            }
        }
        else if (layer->type == "ELU")
        {
            ncnn::ELU *op = (ncnn::ELU *)layer;
            ncnn::ELU *op_default = (ncnn::ELU *)layer_default;

            fprintf_param_value(" 0=%f", alpha)
        }
        else if (layer->type == "Exp")
        {
            ncnn::Exp *op = (ncnn::Exp *)layer;
            ncnn::Exp *op_default = (ncnn::Exp *)layer_default;

            fprintf_param_value(" 0=%f", base)
                fprintf_param_value(" 1=%f", scale)
                    fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "Flatten")
        {
            ncnn::Flatten *op = (ncnn::Flatten *)layer;
            ncnn::Flatten *op_default = (ncnn::Flatten *)layer_default;
            fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "InnerProduct")
        {
            ncnn::InnerProduct *op = (ncnn::InnerProduct *)layer;
            ncnn::InnerProduct *op_default = (ncnn::InnerProduct *)layer_default;

            fprintf_param_value(" 0=%d", num_output)
                fprintf_param_value(" 1=%d", bias_term)
                    fprintf_param_value(" 2=%d", weight_data_size)
                        fprintf_param_value(" 8=%d", int8_scale_term)
                            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty())
                    fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {
                // std::vector<float> weight_int8scale;
                // std::vector<float> blob_int8scale;

                // char key[256];
                // sprintf(key, "%s_param_0", layers[i]->name.c_str());

                // if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                // {
                //     weight_int8scale = weight_int8scale_table[std::string(key)];
                // }

                // if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                // {
                //     blob_int8scale = blob_int8scale_table[layer->name];
                // }

                // // write int8_scale data
                // fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                // fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
                std::vector<int> int_scales;
                int right_shift;
                if (int_result_blob_int8scale_table.find(layer->name) != int_result_blob_int8scale_table.end())
                {
                    int_scales = int_result_blob_int8scale_table[layer->name];
                }

                // if (top_scale_right_shift.find(layer->name) != top_scale_right_shift.end())
                // {
                //     right_shift = top_scale_right_shift[layer->name];
                // }
                // write int8_scale data and shift
                fwrite(int_scales.data(), sizeof(int), int_scales.size(), bp);
                // fwrite(&right_shift, sizeof(int), 1, bp);
            }
        }
        else if (layer->type == "Input")
        {
            ncnn::Input *op = (ncnn::Input *)layer;
            ncnn::Input *op_default = (ncnn::Input *)layer_default;

            fprintf_param_value(" 0=%d", w)
                fprintf_param_value(" 1=%d", h)
                    fprintf_param_value(" 2=%d", c)
                        fprintf(pp, " 8=%d", 1);
            std::vector<int> int_scales;
            // int right_shift;
            if (int_result_blob_int8scale_table.find(layer->name) != int_result_blob_int8scale_table.end())
            {
                int_scales = int_result_blob_int8scale_table[layer->name];
            }
            // if (top_scale_right_shift.find(layer->name) != top_scale_right_shift.end())
            // {
            //     right_shift = top_scale_right_shift[layer->name];
            // }
            // write int8_scale data and shift
            fwrite(int_scales.data(), sizeof(int), int_scales.size(), bp);
            // fwrite(&right_shift, sizeof(int), 1, bp);
        }
        else if (layer->type == "InstanceNorm")
        {
            ncnn::InstanceNorm *op = (ncnn::InstanceNorm *)layer;
            ncnn::InstanceNorm *op_default = (ncnn::InstanceNorm *)layer_default;

            fprintf_param_value(" 0=%d", channels)
                fprintf_param_value(" 1=%f", eps)
        }
        else if (layer->type == "Interp")
        {
            ncnn::Interp *op = (ncnn::Interp *)layer;
            ncnn::Interp *op_default = (ncnn::Interp *)layer_default;

            fprintf_param_value(" 0=%d", resize_type)
                fprintf_param_value(" 1=%f", height_scale)
                    fprintf_param_value(" 2=%f", width_scale)
                        fprintf_param_value(" 3=%d", output_height)
                            fprintf_param_value(" 4=%d", output_width)
        }
        else if (layer->type == "Log")
        {
            ncnn::Log *op = (ncnn::Log *)layer;
            ncnn::Log *op_default = (ncnn::Log *)layer_default;

            fprintf_param_value(" 0=%f", base)
                fprintf_param_value(" 1=%f", scale)
                    fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "LRN")
        {
            ncnn::LRN *op = (ncnn::LRN *)layer;
            ncnn::LRN *op_default = (ncnn::LRN *)layer_default;

            fprintf_param_value(" 0=%d", region_type)
                fprintf_param_value(" 1=%d", local_size)
                    fprintf_param_value(" 2=%f", alpha)
                        fprintf_param_value(" 3=%f", beta)
                            fprintf_param_value(" 4=%f", bias)
        }
        else if (layer->type == "MemoryData")
        {
            ncnn::MemoryData *op = (ncnn::MemoryData *)layer;
            ncnn::MemoryData *op_default = (ncnn::MemoryData *)layer_default;

            fprintf_param_value(" 0=%d", w)
                fprintf_param_value(" 1=%d", h)
                    fprintf_param_value(" 2=%d", c)
                        fwrite_weight_data(op->data, bp);
        }
        else if (layer->type == "MVN")
        {
            ncnn::MVN *op = (ncnn::MVN *)layer;
            ncnn::MVN *op_default = (ncnn::MVN *)layer_default;

            fprintf_param_value(" 0=%d", normalize_variance)
                fprintf_param_value(" 1=%d", across_channels)
                    fprintf_param_value(" 2=%f", eps)
        }
        else if (layer->type == "Normalize")
        {
            ncnn::Normalize *op = (ncnn::Normalize *)layer;
            ncnn::Normalize *op_default = (ncnn::Normalize *)layer_default;

            fprintf_param_value(" 0=%d", across_spatial)
                fprintf_param_value(" 1=%d", channel_shared)
                    fprintf_param_value(" 2=%f", eps)
                        fprintf_param_value(" 3=%d", scale_data_size)
                            fprintf_param_value(" 4=%d", across_channel)

                                fwrite_weight_data(op->scale_data, bp);
        }
        else if (layer->type == "Padding")
        {
            ncnn::Padding *op = (ncnn::Padding *)layer;
            ncnn::Padding *op_default = (ncnn::Padding *)layer_default;

            fprintf_param_value(" 0=%d", top)
                fprintf_param_value(" 1=%d", bottom)
                    fprintf_param_value(" 2=%d", left)
                        fprintf_param_value(" 3=%d", right)
                            fprintf_param_value(" 4=%d", type)
                                fprintf_param_value(" 5=%f", value)
                                    fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Permute")
        {
            ncnn::Permute *op = (ncnn::Permute *)layer;
            ncnn::Permute *op_default = (ncnn::Permute *)layer_default;

            fprintf_param_value(" 0=%d", order_type)
                fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "PixelShuffle")
        {
            ncnn::PixelShuffle *op = (ncnn::PixelShuffle *)layer;
            ncnn::PixelShuffle *op_default = (ncnn::PixelShuffle *)layer_default;

            fprintf_param_value(" 0=%d", upscale_factor)
        }
        else if (layer->type == "Pooling")
        {
            ncnn::Pooling *op = (ncnn::Pooling *)layer;
            ncnn::Pooling *op_default = (ncnn::Pooling *)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
                fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w)
                    fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", stride_w)
            {
                if (op->stride_h != op->stride_w)
                    fprintf(pp, " 12=%d", op->stride_h);
            }
            fprintf_param_value(" 3=%d", pad_left)
            {
                if (op->pad_top != op->pad_left)
                    fprintf(pp, " 13=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left)
                    fprintf(pp, " 14=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top)
                    fprintf(pp, " 15=%d", op->pad_bottom);
            }
            fprintf_param_value(" 4=%d", global_pooling)
                fprintf_param_value(" 5=%d", pad_mode)
                    fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Power")
        {
            ncnn::Power *op = (ncnn::Power *)layer;
            ncnn::Power *op_default = (ncnn::Power *)layer_default;

            fprintf_param_value(" 0=%f", power)
                fprintf_param_value(" 1=%f", scale)
                    fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "PReLU")
        {
            ncnn::PReLU *op = (ncnn::PReLU *)layer;
            ncnn::PReLU *op_default = (ncnn::PReLU *)layer_default;

            fprintf_param_value(" 0=%d", num_slope)

                fwrite_weight_data(op->slope_data, bp);
        }
        else if (layer->type == "PriorBox")
        {
            ncnn::PriorBox *op = (ncnn::PriorBox *)layer;
            ncnn::PriorBox *op_default = (ncnn::PriorBox *)layer_default;

            {
                if (!op->min_sizes.empty())
                    fprintf_param_float_array(0, op->min_sizes, pp);
            }
            {
                if (!op->max_sizes.empty())
                    fprintf_param_float_array(1, op->max_sizes, pp);
            }
            {
                if (!op->aspect_ratios.empty())
                    fprintf_param_float_array(2, op->aspect_ratios, pp);
            }
            fprintf_param_value(" 3=%f", variances[0])
                fprintf_param_value(" 4=%f", variances[1])
                    fprintf_param_value(" 5=%f", variances[2])
                        fprintf_param_value(" 6=%f", variances[3])
                            fprintf_param_value(" 7=%d", flip)
                                fprintf_param_value(" 8=%d", clip)
                                    fprintf_param_value(" 9=%d", image_width)
                                        fprintf_param_value(" 10=%d", image_height)
                                            fprintf_param_value(" 11=%f", step_width)
                                                fprintf_param_value(" 12=%f", step_height)
                                                    fprintf_param_value(" 13=%f", offset)
        }
        else if (layer->type == "Proposal")
        {
            ncnn::Proposal *op = (ncnn::Proposal *)layer;
            ncnn::Proposal *op_default = (ncnn::Proposal *)layer_default;

            fprintf_param_value(" 0=%d", feat_stride)
                fprintf_param_value(" 1=%d", base_size)
                    fprintf_param_value(" 2=%d", pre_nms_topN)
                        fprintf_param_value(" 3=%d", after_nms_topN)
                            fprintf_param_value(" 4=%f", nms_thresh)
                                fprintf_param_value(" 5=%d", min_size)
        }
        else if (layer->type == "PSROIPooling")
        {
            ncnn::PSROIPooling *op = (ncnn::PSROIPooling *)layer;
            ncnn::PSROIPooling *op_default = (ncnn::PSROIPooling *)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
                fprintf_param_value(" 1=%d", pooled_height)
                    fprintf_param_value(" 2=%f", spatial_scale)
                        fprintf_param_value(" 3=%d", output_dim)
        }
        else if (layer->type == "Quantize")
        {
            ncnn::Quantize *op = (ncnn::Quantize *)layer;
            ncnn::Quantize *op_default = (ncnn::Quantize *)layer_default;

            fprintf_param_value(" 0=%f", scale)
        }
        else if (layer->type == "Reduction")
        {
            ncnn::Reduction *op = (ncnn::Reduction *)layer;
            ncnn::Reduction *op_default = (ncnn::Reduction *)layer_default;

            fprintf_param_value(" 0=%d", operation)
                fprintf_param_value(" 1=%d", reduce_all)
                    fprintf_param_value(" 2=%f", coeff)
            {
                if (!op->axes.empty())
                    fprintf_param_int_array(3, op->axes, pp);
            }
            fprintf_param_value(" 4=%d", keepdims)
        }
        else if (layer->type == "ReLU")
        {
            ncnn::ReLU *op = (ncnn::ReLU *)layer;
            ncnn::ReLU *op_default = (ncnn::ReLU *)layer_default;

            fprintf_param_value(" 0=%f", slope)
                fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Reorg")
        {
            ncnn::Reorg *op = (ncnn::Reorg *)layer;
            ncnn::Reorg *op_default = (ncnn::Reorg *)layer_default;

            fprintf_param_value(" 0=%d", stride)
        }
        else if (layer->type == "Requantize")
        {
            ncnn::Requantize *op = (ncnn::Requantize *)layer;
            ncnn::Requantize *op_default = (ncnn::Requantize *)layer_default;

            fprintf_param_value(" 0=%f", scale_in)
                fprintf_param_value(" 1=%f", scale_out)
                    fprintf_param_value(" 2=%d", bias_term)
                        fprintf_param_value(" 3=%d", bias_data_size)
                            fprintf_param_value(" 4=%d", fusion_relu)
        }
        else if (layer->type == "Reshape")
        {
            ncnn::Reshape *op = (ncnn::Reshape *)layer;
            ncnn::Reshape *op_default = (ncnn::Reshape *)layer_default;

            fprintf_param_value(" 0=%d", w)
                fprintf_param_value(" 1=%d", h)
                    fprintf_param_value(" 2=%d", c)
                        fprintf_param_value(" 3=%d", permute)
                            fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "ROIAlign")
        {
            ncnn::ROIAlign *op = (ncnn::ROIAlign *)layer;
            ncnn::ROIAlign *op_default = (ncnn::ROIAlign *)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
                fprintf_param_value(" 1=%d", pooled_height)
                    fprintf_param_value(" 2=%f", spatial_scale)
        }
        else if (layer->type == "ROIPooling")
        {
            ncnn::ROIPooling *op = (ncnn::ROIPooling *)layer;
            ncnn::ROIPooling *op_default = (ncnn::ROIPooling *)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
                fprintf_param_value(" 1=%d", pooled_height)
                    fprintf_param_value(" 2=%f", spatial_scale)
        }
        else if (layer->type == "Scale")
        {
            ncnn::Scale *op = (ncnn::Scale *)layer;
            ncnn::Scale *op_default = (ncnn::Scale *)layer_default;

            fprintf_param_value(" 0=%d", scale_data_size)
                fprintf_param_value(" 1=%d", bias_term)

                    fwrite_weight_data(op->scale_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "ShuffleChannel")
        {
            ncnn::ShuffleChannel *op = (ncnn::ShuffleChannel *)layer;
            ncnn::ShuffleChannel *op_default = (ncnn::ShuffleChannel *)layer_default;

            fprintf_param_value(" 0=%d", group)
        }
        else if (layer->type == "Slice")
        {
            ncnn::Slice *op = (ncnn::Slice *)layer;
            ncnn::Slice *op_default = (ncnn::Slice *)layer_default;

            {
                if (!op->slices.empty())
                    fprintf_param_int_array(0, op->slices, pp);
            }
            fprintf_param_value(" 1=%d", axis)
                fprintf(pp, " 8=%d", 1);
        }
        else if (layer->type == "Softmax")
        {
            ncnn::Softmax *op = (ncnn::Softmax *)layer;
            ncnn::Softmax *op_default = (ncnn::Softmax *)layer_default;

            fprintf_param_value(" 0=%d", axis)

                // HACK
                if (op->axis != 0)
            {
                int fixbug0 = 1;
                fprintf(pp, " 1=%d", fixbug0);
            }
        }
        else if (layer->type == "Split")
        {
            fprintf(pp, " 8=%d", 1);
            if (split_factor_table.count(layer->name) == 1)
            {
                std::vector<int> factors = split_factor_table[layer->name];
                fwrite(factors.data(), sizeof(int), factors.size(), bp);
            }
        }
        else if (layer->type == "Threshold")
        {
            ncnn::Threshold *op = (ncnn::Threshold *)layer;
            ncnn::Threshold *op_default = (ncnn::Threshold *)layer_default;

            fprintf_param_value(" 0=%f", threshold)
        }
        else if (layer->type == "UnaryOp")
        {
            ncnn::UnaryOp *op = (ncnn::UnaryOp *)layer;
            ncnn::UnaryOp *op_default = (ncnn::UnaryOp *)layer_default;

            fprintf_param_value(" 0=%d", op_type)
        }
        else if (layer->type == "YoloDetectionOutput")
        {
            ncnn::YoloDetectionOutput *op = (ncnn::YoloDetectionOutput *)layer;
            ncnn::YoloDetectionOutput *op_default = (ncnn::YoloDetectionOutput *)layer_default;

            fprintf_param_value(" 0=%d", num_class)
                fprintf_param_value(" 1=%d", num_box)
                    fprintf_param_value(" 2=%f", confidence_threshold)
                        fprintf_param_value(" 3=%f", nms_threshold)
            {
                if (!op->biases.empty())
                    fprintf_param_float_array(4, op->biases, pp);
            }
        }
        else if (layer->type == "Yolov3DetectionOutput")
        {
            ncnn::Yolov3DetectionOutput *op = (ncnn::Yolov3DetectionOutput *)layer;
            ncnn::Yolov3DetectionOutput *op_default = (ncnn::Yolov3DetectionOutput *)layer_default;

            fprintf_param_value(" 0=%d", num_class)
                fprintf_param_value(" 1=%d", num_box)
                    fprintf_param_value(" 2=%f", confidence_threshold)
                        fprintf_param_value(" 3=%f", nms_threshold)
            {
                if (!op->biases.empty())
                    fprintf_param_float_array(4, op->biases, pp);
            }
            {
                if (!op->mask.empty())
                    fprintf_param_int_array(5, op->mask, pp);
            }
            {
                if (!op->anchors_scale.empty())
                    fprintf_param_float_array(6, op->anchors_scale, pp);
            }
        }

#undef fprintf_param_value

        fprintf(pp, "\n");

        delete layer_default;
    }

    fclose(pp);
    fclose(bp);

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [calibration table]\n", argv[0]);
        return -1;
    }

    const char *inparam = argv[1];
    const char *inbin = argv[2];
    const char *outparam = argv[3];
    const char *outbin = argv[4];
    const char *int8scale_table_path = argv[5];

    NetQuantize quantizer;

    // parse the calibration scale table
    if (int8scale_table_path)
    {
        bool s2 = read_int8scale_table(int8scale_table_path, quantizer.blob_int8scale_table, quantizer.weight_int8scale_table);
        if (!s2)
        {
            fprintf(stderr, "read_int8scale_table failed\n");
            return -1;
        }
    }

    quantizer.load_param(inparam);
    quantizer.load_model(inbin);

    // Find top scale
    quantizer.find_top_scale();
    std::map<std::string, std::vector<float>>::iterator iter;
    std::map<std::string, std::vector<float>>::iterator iter_bottom = quantizer.blob_int8scale_table.begin();
    std::map<std::string, std::vector<float>>::iterator iter_weight = quantizer.weight_int8scale_table.begin();

    for (iter = quantizer.top_blob_int8scale_table.begin(); iter != quantizer.top_blob_int8scale_table.end();)
    {
        std::string input = "input";
        std::string name = iter->first;
        std::vector<float> top_scales = iter->second;
        std::vector<float> result_scales;
        if (name.find(input) != std::string::npos)
        {
            result_scales.push_back(top_scales[0]);
            quantizer.result_blob_int8scale_table[iter->first] = result_scales;
            iter++;
        }
        else
        {
            std::vector<float> bottom_scales = iter_bottom->second;
            std::vector<float> weight_scales = iter_weight->second;

            int size = top_scales.size();
            if (size == 1)
            {
                for (int i = 0; i < weight_scales.size(); i++)
                {
                    float result_scale = top_scales[0] / (weight_scales[i] * bottom_scales[0]);
                    result_scales.push_back(result_scale);
                }
                quantizer.result_blob_int8scale_table[name] = result_scales;
            }
            else if (size == 2)
            {
                // for (int i = 0; i < weight_scales.size() / 2; i++)
                // {
                //     float result_scale = top_scales[0] / (weight_scales[i] * bottom_scales[0]);
                //     result_scales.push_back(result_scale);
                // }

                // for (int i = weight_scales.size() / 2; i < weight_scales.size(); i++)
                // {
                //     float result_scale = top_scales[1] / (weight_scales[i] * bottom_scales[0]);
                //     result_scales.push_back(result_scale);
                // }
                // TODO for mbv2: split will cause 2 top_scale condition
                // for (int i = 0; i < weight_scales.size(); i++)
                // {
                //     float result_scale = top_scales[0] / (weight_scales[i] * bottom_scales[0]);
                //     result_scales.push_back(result_scale);
                // }

                // for (int i = 0; i < weight_scales.size(); i++)
                // {
                //     float result_scale = top_scales[1] / (weight_scales[i] * bottom_scales[0]);
                //     result_scales.push_back(result_scale);
                // }

                float scale = top_scales[0] > top_scales[1] ? top_scales[0] : top_scales[1];
                for (int i = 0; i < weight_scales.size(); i++)
                {
                    float result_scale = scale / (weight_scales[i] * bottom_scales[0]);
                    result_scales.push_back(result_scale);
                }
            }
            else
            {
                for (int i = 0; i < weight_scales.size(); i++)
                {
                    float result_scale = top_scales[i] / (weight_scales[i] * bottom_scales[0]);
                    result_scales.push_back(result_scale);
                }
                quantizer.result_blob_int8scale_table[name] = result_scales;
            }
            iter++, iter_bottom++, iter_weight++;
        }

        std::vector<int> int_scales;
        int_scales.resize(result_scales.size() + 1);
        int BitN = 7;
        int N = vector_float2int(&int_scales[0], &result_scales[0], result_scales.size(), BitN);
        int_scales[result_scales.size()] = N;
        quantizer.int_result_blob_int8scale_table[name] = int_scales;
        fprintf(stdout, "%s-%d ", name.c_str(), N);
        // for (int i = 0; i < int_scales.size(); i++)
        // {
        //     fprintf(stdout, "%d ", int_scales[i]);
        // }
        // fprintf(stdout, "\n");

        // fprintf(stdout, "#######################################\n");

        // for (int i = 0; i < top_scales.size(); i++)
        // {
        //     fprintf(stdout, "%f ", top_scales[i]);
        // }
        // fprintf(stdout, "\n");

        for (int i = 0; i < result_scales.size(); i++)
        {
            fprintf(stdout, "%f ", result_scales[i]);
        }
        fprintf(stdout, "\n");
        // fprintf(stdout, "#######################################\n");

        // for (int i = 0; i < weight_scales.size(); i++)
        // {
        //     fprintf(stdout, "%f ", weight_scales[i]);
        // }
        // fprintf(stdout, "\n");
    }

    std::map<std::string, std::vector<float>>::iterator top_scale_iter;
    for (top_scale_iter = quantizer.top_blob_int8scale_table.begin(); top_scale_iter != quantizer.top_blob_int8scale_table.end(); top_scale_iter++)
    {
        std::string name = top_scale_iter->first;
        std::vector<float> top_scales = top_scale_iter->second;
        if (top_scales.size() == 2)
        {
            quantizer.split_scale(name, top_scales);
        }
    }

    std::map<std::string, std::vector<float>>::iterator top_scale_float;
    for (top_scale_float = quantizer.top_blob_int8scale_table.begin(); top_scale_float != quantizer.top_blob_int8scale_table.end(); top_scale_float++)
    {
        std::string name = top_scale_float->first;
        std::vector<float> top_scales = top_scale_float->second;
        int BitN = 7;
        std::vector<int> int_scales;
        int_scales.resize(top_scales.size() + 1);
        int N = vector_float2int(&int_scales[0], &top_scales[0], top_scales.size(), BitN);
        fprintf(stdout, "kkkkk %d - %d\n", N, top_scales.size());
        int_scales[top_scales.size()] = N;
        quantizer.int_top_blob_int8scale_table[name] = int_scales;
    }

    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();

    quantizer.save(outparam, outbin);

    return 0;
}
