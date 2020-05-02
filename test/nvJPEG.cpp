/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This sample needs at least CUDA 10.0. It demonstrates usages of the nvJPEG
// library nvJPEG supports single and multiple image(batched) decode. Multiple
// images can be decoded using the API for batch mode

#include "nvjpeg.h"
#include <cuda_runtime_api.h>
#include <string>
#include "cuda_runtime.h"

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

int dev_malloc(void **p, size_t s) { return (int) cudaMalloc(p, s); }

int dev_free(void *p) { return (int) cudaFree(p); }

struct decode_params_t {
    std::string input_dir;
    int batch_size;
    int total_images;
    int dev;
    int warmup;

    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t nvjpeg_handle;
    cudaStream_t stream;

    nvjpegOutputFormat_t fmt;
    bool write_decoded;
    std::string output_dir;

    bool pipelined;
    bool batched;
};

int main(int argc, const char *argv[]) {

    decode_params_t params;
    params.dev = 0;

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));

    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n", params.dev, props.name,
           props.multiProcessorCount, props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &params.nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));
    checkCudaErrors(
        nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, params.batch_size, 1, params.fmt));

    // read source images
    FileNames image_names;
    readInput(params.input_dir, image_names);

    if (params.total_images == -1) {
        params.total_images = image_names.size();
    } else if (params.total_images % params.batch_size) {
        params.total_images = ((params.total_images) / params.batch_size) * params.batch_size;
        std::cout << "Changing total_images number to " << params.total_images << " to be multiple of batch_size - "
                  << params.batch_size << std::endl;
    }

    std::cout << "Decoding images in directory: " << params.input_dir << ", total " << params.total_images
              << ", batchsize " << params.batch_size << std::endl;

    double total;
    if (process_images(image_names, params, total))
        return EXIT_FAILURE;
    std::cout << "Total decoding time: " << total << std::endl;
    std::cout << "Avg decoding time per image: " << total / params.total_images << std::endl;
    std::cout << "Avg images per sec: " << params.total_images / total << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total / ((params.total_images + params.batch_size - 1) / params.batch_size) << std::endl;

    checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));

    return EXIT_SUCCESS;
}
