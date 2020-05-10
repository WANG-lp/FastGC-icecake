#include "nvjpeg.h"
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <string>
#include <vector>
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
    // realloc output buffer if required
    for (int c = 0; c < nCom; c++) {
        int aw = widths[c];
        int ah = heights[c];
        int sz = aw * ah;
        out_img_t.pitch[c] = aw;
        checkCudaErrors(cudaMalloc(&out_img_t.channel[c], sz));
    }

    checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state, data.data(), data.size(), NVJPEG_OUTPUT_RGB,
                                 &out_img_t, params.stream));
    checkCudaErrors(cudaStreamSynchronize(params.stream));
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

    decode_image(image_data);

    // checkCudaErrors(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));
    // checkCudaErrors(
    //     nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, params.batch_size, 1, params.fmt));

    // checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));
    return 0;
}