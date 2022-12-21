/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;
using DT = uint8_t;

template<typename T>
void print(std::string name, std::vector<T> data) {
    std::cout << name << ": ";
    for (const T& val : data) {
        std::cout.operator<<(val);
        std::cout << " ";
    }
    std::cout << "\n";
}

template<typename T>
void print(std::string name, T val) {
    std::cout << name << ": " << std::to_string(val) << "\n";
}

template<typename T>
auto onednn_dtype() {
    if (std::is_same<DT, float>::value)
        return dt::f32;
    if (std::is_same<DT, uint8_t>::value)
        return dt::u8;
    if (std::is_same<DT, int8_t>::value)
        return dt::s8;
    throw std::invalid_argument("Unexpected dtype");
}

void softmax_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 10; // channels

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC};

    // Allocate buffer.
    std::vector<float> usr_src_data(product(src_dims));
    std::vector<DT> src_data(usr_src_data.size());

    std::generate(usr_src_data.begin(), usr_src_data.end(), []() {
        static int i = 0;
        float v =  std::cos(i++ / 10.f);
        return v;
    });
    print("usr_src_data", usr_src_data);
    float max_usr_src_data = *std::max_element(usr_src_data.begin(), usr_src_data.end());
    print("max_usr_src_data", max_usr_src_data);

    if (!std::is_same<DT, float>::value) {
        const float scale = static_cast<float>(std::numeric_limits<DT>::max()) / max_usr_src_data;
        std::transform(usr_src_data.cbegin(), usr_src_data.cend(), src_data.begin(), [scale](float value){return value*scale;});
    } else {
        std::copy(usr_src_data.begin(), usr_src_data.end(), src_data.begin());
    }

    print("src_data", src_data);

    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(src_dims, onednn_dtype<DT>(), tag::nc);
    auto src_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Softmax axis.
    const int axis = 1;

    // Create operation descriptor.
    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_inference, src_md, axis);

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {127.0f});
    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, attr, engine);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
    print("output", src_data);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            softmax_example, parse_engine_kind(argc, argv));
}
