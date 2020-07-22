#include "nvjpeg.h"
#include <cuda_runtime_api.h>
#include <omp.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <string>
#include <vector>
#include "../include/jpeg_decoder.hpp"
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
void get_height_width(const vector<uint8_t> &data, vector<int> &width, vector<int> &height) {
    int nCom;
    nvjpegChromaSubsampling_t subsampling;
    checkCudaErrors(nvjpegGetImageInfo(params.nvjpeg_handle, data.data(), data.size(), &nCom, &subsampling,
                                       width.data(), height.data()));
}
void decode_batched(const vector<vector<uint8_t>> &data, int num_threads, int max_iter) {
    spdlog::info("decode_batched:");

    vector<const uint8_t *> data_raw;
    vector<size_t> lengths;
    data_raw.resize(data.size());
    lengths.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        data_raw[i] = data[i].data();
        lengths[i] = data[i].size();
    }

    vector<nvjpegImage_t> out_images;
    out_images.resize(data.size());
    vector<vector<int>> widths;
    vector<vector<int>> heights;
    widths.resize(data.size());
    heights.resize(data.size());

    for (int i = 0; i < data.size(); i++) {
        widths[i].resize(4);
        heights[i].resize(4);
        get_height_width(data[i], widths[i], heights[i]);

        for (int c = 0; c < 4; c++) {
            int aw = widths[i][c];
            int ah = heights[i][c];
            int sz = aw * ah;
            out_images[i].pitch[c] = aw;
            checkCudaErrors(cudaMalloc(&out_images[i].channel[c], sz));
        }
    }
    spdlog::info("width: {}, height: {}", widths[0][0], heights[0][0]);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    checkCudaErrors(nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, data.size(), num_threads,
                                                  NVJPEG_OUTPUT_RGB));
    cudaEventRecord(start, params.stream);
    for (int iter = 0; iter < max_iter; iter++) {
// checkCudaErrors(nvjpegDecodeBatched(params.nvjpeg_handle, params.nvjpeg_state, data_raw.data(), lengths.data(),
// out_images.data(), params.stream));
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < data.size(); i++) {
            checkCudaErrors(nvjpegDecodeBatchedPhaseOne(params.nvjpeg_handle, params.nvjpeg_state, data[i].data(),
                                                        data[i].size(), i, omp_get_thread_num(), params.stream));
        }
        checkCudaErrors(nvjpegDecodeBatchedPhaseTwo(params.nvjpeg_handle, params.nvjpeg_state, params.stream));
        checkCudaErrors(
            nvjpegDecodeBatchedPhaseThree(params.nvjpeg_handle, params.nvjpeg_state, out_images.data(), params.stream));
    }

    cudaEventRecord(stop, params.stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("DecodeBatched time {}ms", milliseconds);
    size_t total_decoded = data.size() * max_iter;
    spdlog::info("Decoded {} images, Decoding speed: {} images/second", total_decoded,
                 total_decoded / (milliseconds / 1000.0));
    // copy_out(out_images[0], widths[0][0], heights[0][0]);
}
void decode_decoupled(vector<uint8_t> data, int batch_size) {
    spdlog::info("decode_decoupled:");

    nvjpegJpegDecoder_t decoder_t;
    // nvjpegJpegState_t decoder_state;
    nvjpegBufferPinned_t pin_buffer;
    nvjpegBufferDevice_t dev_buffer;
    nvjpegDecodeParams_t decode_params;

    checkCudaErrors(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_HYBRID, &decoder_t));
    // checkCudaErrors(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_GPU_HYBRID, &decoder_t));

    checkCudaErrors(nvjpegDecoderStateCreate(params.nvjpeg_handle, decoder_t, &params.nvjpeg_state));
    checkCudaErrors(nvjpegBufferPinnedCreate(params.nvjpeg_handle, nullptr, &pin_buffer));
    checkCudaErrors(nvjpegBufferDeviceCreate(params.nvjpeg_handle, nullptr, &dev_buffer));
    checkCudaErrors(nvjpegStateAttachPinnedBuffer(params.nvjpeg_state, pin_buffer));
    checkCudaErrors(nvjpegStateAttachDeviceBuffer(params.nvjpeg_state, dev_buffer));
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

    cudaEventRecord(start, params.stream);
    for (int i = 0; i < batch_size; i++) {
        checkCudaErrors(
            nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder_t, params.nvjpeg_state, decode_params, jpeg_stream));
        // cudaEventRecord(stop, params.stream);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // spdlog::info("decodeJpegHost time {}ms", milliseconds);

        // cudaEventRecord(start, params.stream);
        // checkCudaErrors(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder_t, params.nvjpeg_state,
        //                                                  jpeg_stream, params.stream));
        // cudaEventRecord(stop, params.stream);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // spdlog::info("transfer time {}ms", milliseconds);

        // cudaEventRecord(start, params.stream);
        // checkCudaErrors(
        //     nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder_t, params.nvjpeg_state, &out_img_t, params.stream));
    }
    cudaEventRecord(stop, params.stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("decodeJpegDevice time {}ms", milliseconds);
    spdlog::info("speed: {} images/sec", batch_size / (milliseconds / 1000.0));
    checkCudaErrors(cudaStreamSynchronize(params.stream));
    copy_out(out_img_t, widths[0], heights[0]);
    exit(0);
}

void decode_image(vector<uint8_t> data) {
    spdlog::info("decode_image:");
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

    cudaEventRecord(start, params.stream);
    checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state, data.data(), data.size(), NVJPEG_OUTPUT_RGB,
                                 &out_img_t, params.stream));
    cudaEventRecord(stop, params.stream);
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
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &params.nvjpeg_handle));  // step 1
    checkCudaErrors(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));            // step 2
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

    assert(argc >= 5);

    int thread_num = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int batch_size = atoi(argv[4]);
    spdlog::info("thread num: {}", thread_num);
    spdlog::info("max iter: {}", max_iter);
    spdlog::info("batch size: {}", batch_size);

    decode_decoupled(image_data, batch_size);

    vector<vector<uint8_t>> data_batched;
    data_batched.resize(batch_size);
    for (int i = 0; i < data_batched.size(); i++) {
        data_batched[i].resize(image_data.size());
        memcpy(data_batched[i].data(), image_data.data(), image_data.size());
    }
    decode_batched(data_batched, thread_num, max_iter);

    checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));
    return 0;
}