#include "nvjpeg.h"
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <string>
#include <vector>
#include "../src/jpeg_decoder.hpp"
#include "cuda_runtime.h"

using std::string;
using std::vector;

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

struct decode_params_t {
    int dev;
    int warmup;

    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t nvjpeg_handle;
    cudaStream_t stream;
};
decode_params_t params;

int dev_malloc(void **p, size_t s) { return (int) cudaMalloc(p, s); }
int dev_free(void *p) { return (int) cudaFree(p); }

void copy_out(nvjpegImage_t &out_img_t, int width, int height) {
    // copy out
    std::vector<unsigned char> vchanR(height * width);
    std::vector<unsigned char> vchanG(height * width);
    std::vector<unsigned char> vchanB(height * width);
    unsigned char *chanR = vchanR.data();
    unsigned char *chanG = vchanG.data();
    unsigned char *chanB = vchanB.data();
    checkCudaErrors(cudaMemcpy2D(chanR, (size_t) width, out_img_t.channel[0], (size_t) out_img_t.pitch[0], width,
                                 height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(chanG, (size_t) width, out_img_t.channel[1], (size_t) out_img_t.pitch[1], width,
                                 height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(chanB, (size_t) width, out_img_t.channel[2], (size_t) out_img_t.pitch[2], width,
                                 height, cudaMemcpyDeviceToHost));
    jpeg_dec::writeBMP("/tmp/out.bmp", chanR, chanG, chanB, width, height);
}
void decode_decoupled(vector<uint8_t> data) {
    nvjpegJpegDecoder_t decoder_t;
    nvjpegJpegState_t decoder_state;
    nvjpegBufferPinned_t pin_buffer;
    nvjpegBufferDevice_t dev_buffer;
    nvjpegDecodeParams_t decode_params;

    checkCudaErrors(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_HYBRID, &decoder_t));
    // checkCudaErrors(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_GPU_HYBRID, &decoder_t));

    checkCudaErrors(nvjpegDecoderStateCreate(params.nvjpeg_handle, decoder_t, &decoder_state));
    checkCudaErrors(nvjpegBufferPinnedCreate(params.nvjpeg_handle, nullptr, &pin_buffer));
    checkCudaErrors(nvjpegBufferDeviceCreate(params.nvjpeg_handle, nullptr, &dev_buffer));
    checkCudaErrors(nvjpegStateAttachPinnedBuffer(decoder_state, pin_buffer));
    checkCudaErrors(nvjpegStateAttachDeviceBuffer(decoder_state, dev_buffer));
    checkCudaErrors(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &decode_params));
    checkCudaErrors(nvjpegDecodeParamsSetOutputFormat(decode_params, NVJPEG_OUTPUT_RGB));

    nvjpegJpegStream_t jpeg_stream;
    checkCudaErrors(nvjpegJpegStreamCreate(params.nvjpeg_handle, &jpeg_stream));
    checkCudaErrors(nvjpegJpegStreamParse(params.nvjpeg_handle, data.data(), data.size(), 1, 0, jpeg_stream));

    nvjpegImage_t out_img_t;
    unsigned int nCom, widths[4], heights[4];
    checkCudaErrors(nvjpegJpegStreamGetComponentsNum(jpeg_stream, &nCom));
    spdlog::info("total channel {}", nCom);
    for (auto i = 0; i < nCom; i++) {
        checkCudaErrors(nvjpegJpegStreamGetComponentDimensions(jpeg_stream, i, &widths[i], &heights[i]));
        spdlog::info("ch:{}, width: {}, height: {}", i, widths[i], heights[i]);
    }
    for (int c = 0; c < nCom; c++) {
        int aw = widths[c];
        int ah = heights[c];
        int sz = aw * ah;
        out_img_t.pitch[c] = aw;
        checkCudaErrors(cudaMalloc(&out_img_t.channel[c], sz));
    }
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    checkCudaErrors(nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder_t, decoder_state, decode_params, jpeg_stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("decodeJpegHost time {}ms", milliseconds);

    cudaEventRecord(start);
    checkCudaErrors(
        nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder_t, decoder_state, jpeg_stream, params.stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("transfer time {}ms", milliseconds);

    cudaEventRecord(start);
    checkCudaErrors(nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder_t, decoder_state, &out_img_t, params.stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("decodeJpegDevice time {}ms", milliseconds);
    checkCudaErrors(cudaStreamSynchronize(params.stream));
    copy_out(out_img_t, widths[0], heights[0]);
}

void decode_image(vector<uint8_t> data) {

    int nCom;
    nvjpegChromaSubsampling_t sampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    checkCudaErrors(nvjpegGetImageInfo(params.nvjpeg_handle, data.data(), data.size(), &nCom, &sampling, widths,
                                       heights));  // step 3
    spdlog::info("total channel {}, sampling: {}", nCom, sampling);
    for (int ch = 0; ch < nCom; ch++) {
        spdlog::info("channel: {}, width: {}, height: {}", ch, widths[ch], heights[ch]);
    }

    nvjpegImage_t out_img_t;
    int width = widths[0];
    int height = heights[0];
    // realloc output buffer if required
    for (int c = 0; c < nCom; c++) {
        int aw = widths[c];
        int ah = heights[c];
        int sz = aw * ah;
        out_img_t.pitch[c] = aw;
        checkCudaErrors(cudaMalloc(&out_img_t.channel[c], sz));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state, data.data(), data.size(), NVJPEG_OUTPUT_RGB,
                                 &out_img_t, params.stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaErrors(cudaStreamSynchronize(params.stream));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("decode time {}ms", milliseconds);
    copy_out(out_img_t, width, height);
}

int main(int argc, char **argv) {

    params.dev = 0;

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));

    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n", params.dev, props.name,
           props.multiProcessorCount, props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(
        nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, nullptr, 0, &params.nvjpeg_handle));  // step 1
    checkCudaErrors(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));              // step 2
    // stream for decoding
    checkCudaErrors(cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

    vector<uint8_t> image_data;
    std::ifstream ifs(argv[1], std::ios::binary | std::ios::ate);
    if (!ifs.good()) {
        spdlog::critical("cannot open file {}", argv[1]);
        exit(1);
    }
    std::streamsize fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    image_data.resize(fsize);
    ifs.read((char *) image_data.data(), fsize);

    decode_decoupled(image_data);

    // checkCudaErrors(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));
    // checkCudaErrors(
    //     nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, params.batch_size, 1, params.fmt));

    // checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));
    return 0;
}