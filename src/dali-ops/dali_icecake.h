#pragma once

#include <omp.h>
#include <iostream>
#include <string>
#include <vector>

#include "../../include/JCache.hpp"
#include "../../include/gpu_decoder.hpp"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"

using namespace dali;

namespace jpegdec {
class DaliIcecake : public dali::Operator<dali::CPUBackend> {
   public:
    inline explicit DaliIcecake(const dali::OpSpec &spec) : ::dali::Operator<::dali::CPUBackend>(spec) {}

    virtual inline ~DaliIcecake() = default;

    DaliIcecake(const DaliIcecake &) = delete;
    DaliIcecake &operator=(const DaliIcecake &) = delete;
    DaliIcecake(DaliIcecake &&) = delete;
    DaliIcecake &operator=(DaliIcecake &&) = delete;

   protected:
    bool CanInferOutputs() const override { return false; }

    bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                   const ::dali::workspace_t<::dali::CPUBackend> &ws) override {
        return false;
    }

    void RunImpl(dali::SampleWorkspace &ws) override {
        auto &output = ws.OutputRef<::dali::CPUBackend>(0);
        ::dali::TensorShape<> list_shape{};
        list_shape.resize(1);
        output.Resize(list_shape);

        int num_input = ws.NumInput();
        printf("num_input: %d\n", num_input);
        fprintf(stderr, "in cpu\n");
        std::cerr << "in cput" << std::endl;
    }

   private:
    int thread_num;
};
class DaliIcecakeMixed : public dali::Operator<dali::MixedBackend> {
   public:
    inline explicit DaliIcecakeMixed(const ::dali::OpSpec &spec) : ::dali::Operator<dali::MixedBackend>(spec) {
        decoder = std::shared_ptr<jpeg_dec::GPUDecoder>(new jpeg_dec::GPUDecoder(batch_size_));
    }

    virtual inline ~DaliIcecakeMixed() {}

    DaliIcecakeMixed(const DaliIcecakeMixed &) = delete;
    DaliIcecakeMixed &operator=(const DaliIcecakeMixed &) = delete;
    DaliIcecakeMixed(DaliIcecakeMixed &&) = delete;
    DaliIcecakeMixed &operator=(DaliIcecakeMixed &&) = delete;

   protected:
    bool CanInferOutputs() const override { return false; }

    bool SetupImpl(std::vector<dali::OutputDesc> &output_desc, const dali::MixedWorkspace &ws) override {
        return false;
    }

    using dali::OperatorBase::Run;
    void Run(::dali::MixedWorkspace &ws) override {
        std::vector<std::vector<Index>> output_shape(batch_size_);
        // Creating output shape and setting the order of images so the largest are processed first
        // (for load balancing)
        std::vector<std::shared_ptr<JPEG_HEADER>> header_ptrs(batch_size_);
        std::vector<std::pair<size_t, size_t>> image_order(batch_size_);
#pragma omp prallel for num_threads(num_threads_)
        for (int i = 0; i < batch_size_; i++) {
            const auto &info_tensor = ws.Input<CPUBackend>(0, i);
            auto data = static_cast<const char *>(info_tensor.raw_data());
            header_ptrs[i] =
                std::shared_ptr<JPEG_HEADER>(jcache::deserialization_header(string(data, data + info_tensor.size())));
            restore_block_offset_from_compact(header_ptrs[i].get());
            int c = 3;
            output_shape[i] = {header_ptrs[i]->height, header_ptrs[i]->width, c};
            image_order[i] = std::make_pair(volume(output_shape[i]), i);
            decoder->do_decode_phase1(i, header_ptrs[i].get());
        }
        // std::sort(image_order.begin(), image_order.end(), std::greater<std::pair<size_t, size_t>>());

        auto &output = ws.Output<GPUBackend>(0);
        output.Resize(output_shape);
        output.SetLayout("HWC");
        TypeInfo type = TypeInfo::Create<uint8_t>();
        output.set_type(type);
        // #pragma omp parallel for num_threads(num_threads_)
        for (size_t idx = 0; idx < image_order.size(); idx++) {
            auto &size_idx = image_order[idx];
            const int sample_idx = size_idx.second;

            const auto &info_tensor = ws.Input<CPUBackend>(0, sample_idx);

            auto *output_data = output.mutable_tensor<uint8_t>(sample_idx);
            decoder->do_decode_phase2(sample_idx, output_data);
        }
    }

   private:
    std::shared_ptr<jpeg_dec::GPUDecoder> decoder;
};

}  // namespace jpegdec
