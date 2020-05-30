#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "../../include/JCache.hpp"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"
#include "libgpujpeg/gpujpeg_decoder.h"

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
    inline explicit DaliIcecakeMixed(const ::dali::OpSpec &spec) : ::dali::Operator<dali::MixedBackend>(spec) {}

    virtual inline ~DaliIcecakeMixed() = default;

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
        for (int i = 0; i < batch_size_; i++) {
            const auto &info_tensor = ws.Input<CPUBackend>(0, i);
            auto data = static_cast<const char *>(info_tensor.raw_data());
            header_ptrs[i] =
                std::shared_ptr<JPEG_HEADER>(jcache::deserialization_header(string(data, data + info_tensor.size())));
            int c = 3;
            output_shape[i] = {header_ptrs[i]->height, header_ptrs[i]->width, c};
            image_order[i] = std::make_pair(volume(output_shape[i]), i);
        }
        std::sort(image_order.begin(), image_order.end(), std::greater<std::pair<size_t, size_t>>());

        auto &output = ws.Output<GPUBackend>(0);
        output.Resize(output_shape);
        output.SetLayout("HWC");
        TypeInfo type = TypeInfo::Create<uint8_t>();
        output.set_type(type);

        for (auto &size_idx : image_order) {
            const int sample_idx = size_idx.second;

            const auto &info_tensor = ws.Input<CPUBackend>(0, sample_idx);
            //     const ImageInfo *info;
            //     const StateNvJPEG *nvjpeg_state;
            //     std::tie(info, nvjpeg_state) = GetInfoState(info_tensor, state_tensor);

            // const auto file_name = info_tensor.GetSourceInfo();

            auto *output_data = output.mutable_tensor<uint8_t>(sample_idx);
            auto decoder = gpujpeg_decoder_create(nullptr);
            gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
            struct gpujpeg_decoder_output decoder_output;
            gpujpeg_decoder_output_set_default(&decoder_output);
            decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
            decoder_output.data = output_data;
            int rc;
            restore_block_offset_from_compact(header_ptrs[sample_idx].get());

            if ((rc = gpujpeg_decoder_decode_phase1(decoder, nullptr, 0, &decoder_output,
                                                    header_ptrs[sample_idx].get())) != 0) {
                fprintf(stderr, "Failed to decode image !\n");
            }

            if ((rc = gpujpeg_decoder_decode_phase2(decoder, &decoder_output)) != 0) {
                fprintf(stderr, "Failed to decode image!\n");
            }

            gpujpeg_decoder_destroy(decoder);

            //     if (info->nvjpeg_support) {
            //         nvjpegImage_t nvjpeg_image;
            //         nvjpeg_image.channel[0] = output_data;
            //         nvjpeg_image.pitch[0] = NumberOfChannels(output_image_type_) * info->widths[0];

            //         nvjpegJpegState_t state = GetNvjpegState(*nvjpeg_state);

            //         NVJPEG_CALL(nvjpegStateAttachDeviceBuffer(state, device_buffer_));

            //         nvjpegJpegDecoder_t decoder = GetDecoder(nvjpeg_state->nvjpeg_backend);
            //         NVJPEG_CALL_EX(
            //             nvjpegDecodeJpegTransferToDevice(handle_, decoder, state, nvjpeg_state->jpeg_stream,
            //             ws.stream()), file_name);

            //         NVJPEG_CALL_EX(nvjpegDecodeJpegDevice(handle_, decoder, state, &nvjpeg_image, ws.stream()),
            //         file_name);
            //     } else {
            //         // Fallback was handled by CPU op and wrote OpenCV ouput in Input #2
            //         // we just need to copy to device
            //         auto &in = ws.Input<CPUBackend>(2, sample_idx);
            //         const auto *input_data = in.data<uint8_t>();
            //         auto *output_data = output.mutable_tensor<uint8_t>(sample_idx);
            //         CUDA_CALL(cudaMemcpyAsync(output_data, input_data,
            //                                   info->heights[0] * info->widths[0] *
            //                                   NumberOfChannels(output_image_type_), cudaMemcpyHostToDevice,
            //                                   ws.stream()));
            //     }
        }
    }
};

}  // namespace jpegdec
