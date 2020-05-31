#include "../include/gpu_decoder.hpp"

#include "../GPUJPEG/libgpujpeg/gpujpeg_common.h"
#include "../GPUJPEG/libgpujpeg/gpujpeg_decoder.h"

namespace jpeg_dec {
GPUDecoder::GPUDecoder(int thread_num) : decoder_idx(0) {
    decoders.resize(thread_num, nullptr);
    for (int i = 0; i < thread_num; i++) {
        decoders[i] = gpujpeg_decoder_create(nullptr);
        gpujpeg_decoder_set_output_format(decoders[i], GPUJPEG_RGB, GPUJPEG_444_U8_P012);
    }
}
GPUDecoder::~GPUDecoder() {
    for (auto& d : decoders) {
        gpujpeg_decoder_destroy(d);
    }
}

int GPUDecoder::do_decode(void* jpeg_header, uint8_t* out_ptr) {
    size_t which_decoder = decoder_idx.fetch_add(1) % decoders.size();
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);
    decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    decoder_output.data = out_ptr;
    int rc = 0;

    if ((rc = gpujpeg_decoder_decode_phase1(decoders[which_decoder], nullptr, 0, jpeg_header)) != 0) {
        fprintf(stderr, "Failed to decode image !\n");
        return rc;
    }

    if ((rc = gpujpeg_decoder_decode_phase2(decoders[which_decoder], &decoder_output)) != 0) {
        fprintf(stderr, "Failed to decode image!\n");
        return rc;
    }

    return rc;
}

int GPUDecoder::do_decode_phase1(size_t which_decode, void* jpeg_header) {
    int rc = 0;
    if ((rc = gpujpeg_decoder_decode_phase1(decoders[which_decode], nullptr, 0, jpeg_header)) != 0) {
        fprintf(stderr, "Failed to decode image !\n");
        return rc;
    }
    return rc;
}

int GPUDecoder::do_decode_phase2(size_t which_decode, uint8_t* out_ptr) {
    int rc = 0;
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);
    decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    decoder_output.data = out_ptr;
    if ((rc = gpujpeg_decoder_decode_phase2(decoders[which_decode], &decoder_output)) != 0) {
        fprintf(stderr, "Failed to decode image!\n");
        return rc;
    }
    return rc;
}

}  // namespace jpeg_dec