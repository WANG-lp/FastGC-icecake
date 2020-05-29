#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <omp.h>
#include "../include/JCache.hpp"
#include "../include/jpeg_decoder_export.h"
#include "../libgpujpeg/gpujpeg.h"
#include "../libgpujpeg/gpujpeg_common.h"
#include "gpujpeg_common_internal.h"   // TIMER
#include "gpujpeg_decoder_internal.h"  // TIMER
#include "gpujpeg_util.h"

#include <cuda_runtime.h>

using std::string;
using std::vector;

int rc;
void dumpImg(const uint8_t* data, size_t data_size, int width, int height) {
    vector<uint8_t> chanR, chanG, chanB;
    chanR.resize(width * height);
    chanG.resize(width * height);
    chanB.resize(width * height);

    int pix_count = 0;
    for (int i = 0; i < data_size;) {
        chanR[pix_count] = data[i];
        chanG[pix_count] = data[i + 1];
        chanB[pix_count] = data[i + 2];
        i += 3;
        pix_count++;
    }
    writeBMP("/tmp/out.bmp", chanR.data(), chanG.data(), chanB.data(), width, height);
}
void decode(vector<vector<uint8_t>> image_data, int num_thread, int max_iter, void* jpeg_header) {
    vector<gpujpeg_decoder*> decoders(image_data.size() + 4);
    vector<struct gpujpeg_decoder_output> decoder_outputs(image_data.size());

    for (int i = 0; i < image_data.size(); i++) {
        decoders[i] = gpujpeg_decoder_create(NULL);
        assert(decoders[i] != nullptr);
        gpujpeg_decoder_output_set_default(&decoder_outputs[i]);
        decoder_outputs[i].type = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
        gpujpeg_decoder_set_output_format(decoders[i], GPUJPEG_RGB, GPUJPEG_444_U8_P012);

        rc = gpujpeg_decoder_decode_phase1(decoders[i], image_data[i].data(), image_data[i].size(), &decoder_outputs[i],
                                           jpeg_header);
        assert(rc == 0);
        // Decode image
        rc = gpujpeg_decoder_decode_phase2(decoders[i], &decoder_outputs[i]);
        assert(rc == 0);  // Decode image
    }

    printf("create ok\n");

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < max_iter; iter++) {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < image_data.size(); i++) {
            // void* jpeg_header = decoders[i % num_thread]->reader->jpeg_header_raw;
            if (jpeg_header && get_jpeg_header_status(jpeg_header) == 1) {
                rc = gpujpeg_decoder_decode_phase1(decoders[i], image_data[i].data(), image_data[i].size(),
                                                   &decoder_outputs[i], jpeg_header);
            } else {
                rc = gpujpeg_decoder_decode_phase1(decoders[i], image_data[i].data(), image_data[i].size(),
                                                   &decoder_outputs[i], nullptr);
            }

            assert(rc == 0);
            // Decode image
            rc = gpujpeg_decoder_decode_phase2(decoders[i], &decoder_outputs[i]);
            assert(rc == 0);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    size_t decoded_imgs = image_data.size() * max_iter;

    printf("decode time: %.5f ms\n", milliseconds);
    printf("decoded images: %d, speed: %f images/second\n", decoded_imgs, decoded_imgs / (milliseconds / 1000.));

    for (int i = 0; i < image_data.size(); i++) {
        // Destroy decoder
        gpujpeg_decoder_destroy(decoders[i]);
    }
}
int warmup(const char* input, gpujpeg_decoder* decoder, uint8_t* image, int image_size, void* jpeg_header, int iter) {
    assert(decoder != nullptr);

    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

    decoder->coder.param.verbose = 0;

    // Decode image
    GPUJPEG_TIMER_INIT();

    // Prepare decoder output buffer
    struct gpujpeg_decoder_output decoder_output;

    gpujpeg_decoder_output_set_default(&decoder_output);
    float ave_duration_time = 0.0;
    float ave_copy_time = 0.0;
    float ave_gpu_time = 0.0;
    float ave_decode_time = 0.0;
    int max_iter = iter;
    for (int i = 0; i < max_iter; i++) {
        GPUJPEG_TIMER_START();
        // Decode image
        if ((rc = gpujpeg_decoder_decode_phase1(decoder, image, image_size, &decoder_output, jpeg_header)) != 0) {
            fprintf(stderr, "Failed to decode image [%s]!\n", input);
            return -1;
        }

        if ((rc = gpujpeg_decoder_decode_phase2(decoder, &decoder_output)) != 0) {
            fprintf(stderr, "Failed to decode image [%s]!\n", input);
            return -1;
        }
        GPUJPEG_TIMER_STOP();
        float duration = GPUJPEG_TIMER_DURATION();
        ave_decode_time += duration;
        ave_duration_time += decoder->coder.duration_stream;
        // ave_copy_time += decoder->coder.
        ave_gpu_time += decoder->coder.duration_in_gpu;
    }
    printf("\nDecoding Image [%s]\n", input);

    printf(" -Stream Reader:     %10.5f ms\n", ave_duration_time / max_iter);
    printf(" -Copy To Device:    %10.2f ms\n", ave_copy_time / max_iter);
    printf("Decode Image GPU:    %10.5f ms (only in-GPU processing)\n", ave_gpu_time / max_iter);
    // printf("Decode Image Bare:   %10.2f ms (without copy to/from GPU memory)\n", duration -
    // decoder->coder.duration_memory_to - decoder->coder.duration_memory_from);
    printf("Decode Image:        %10.5f ms\n\n", ave_decode_time / max_iter);

    dumpImg(decoder_output.data, decoder_output.data_size, decoder->coder.param_image.width,
            decoder->coder.param_image.height);
    printf("Decompressed Size:   %10.d bytes\n", decoder_output.data_size);

    return 0;
}

size_t get_filesize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

int RPC_test(const string& fname, uint8_t* image, size_t len) {
    jcache::JPEGCacheClient jcacheclient("127.0.0.1", 8090);
    int ret = jcacheclient.put("image1.jpeg", image, len);
    assert(ret == 0);
    printf("put image ok\n");
    // auto header = jcacheclient.get("image1.jpeg");
    auto header_ptr = jcacheclient.getWithROI("image1.jpeg", 0, 0, 224, 224);
    restore_block_offset_from_compact(header_ptr);
    auto fsize = get_filesize(fname.c_str());
    printf("file size: %f\n", fsize / 1024.0);
    auto& header = *header_ptr;

    // jcache::JCache jc;
    // jc.putJPEG(image, len, "image1.jpeg");
    // auto header_ptr = jc.getHeaderwithCrop("image1.jpeg", 0, 0, 224, 224);
    // // auto header_ptr = jc.getHeader("image1.jpeg");
    // auto& header = *header_ptr;
    // restore_block_offset_from_compact(header_ptr);

    for (int i = 0; i < 10; i++) {
        printf("%d, %d, %d\n", header.blockpos[i].byte_offset, header.blockpos[i].bit_offset,
               header.blockpos[i].dc_value);
    }

    gpujpeg_decoder* decoder1 = gpujpeg_decoder_create(0);
    gpujpeg_decoder_set_output_format(decoder1, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);

    if ((rc = gpujpeg_decoder_decode_phase1(decoder1, nullptr, 0, &decoder_output, &header)) != 0) {
        fprintf(stderr, "Failed to decode image !\n");
        return -1;
    }

    if ((rc = gpujpeg_decoder_decode_phase2(decoder1, &decoder_output)) != 0) {
        fprintf(stderr, "Failed to decode image!\n");
        return -1;
    }
    printf("width: %d, height: %d\n", decoder1->coder.param_image.width, decoder1->coder.param_image.height);
    dumpImg(decoder_output.data, decoder_output.data_size, decoder1->coder.param_image.width,
            decoder1->coder.param_image.height);

    gpujpeg_decoder_destroy(decoder1);
    return 0;
}

int main(int argc, char** argv) {
    if (gpujpeg_init_device(0, GPUJPEG_VERBOSE) != 0)
        return -1;

    // Decode images

    // Get and check input and output image
    const char* input = argv[1];
    const char* output = argv[2];

    enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(input);
    enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(output);
    if (input_format != GPUJPEG_IMAGE_FILE_JPEG) {
        fprintf(stderr, "Decoder input file [%s] should be JPEG image (*.jpg)!\n", input);
        return -1;
    }
    if ((output_format & GPUJPEG_IMAGE_FILE_RAW) == 0) {
        fprintf(stderr, "[Warning] Decoder output file [%s] should be raw image (*.rgb, *.yuv, *.r)!\n", output);
        if (output_format & GPUJPEG_IMAGE_FILE_JPEG) {
            return -1;
        }
    }

    // Load image
    int image_size = 0;
    uint8_t* image = NULL;
    if (gpujpeg_image_load_from_file(input, &image, &image_size) != 0) {
        fprintf(stderr, "Failed to load image [%s]!\n", input);
        return -1;
    }

    return RPC_test(input, image, image_size);

    gpujpeg_decoder* decoder1 = gpujpeg_decoder_create(0);
    gpujpeg_decoder* decoder2 = gpujpeg_decoder_create(0);

    warmup(input, decoder1, image, image_size, nullptr, 1);
    printf("\n");
    void* jpeg_header_raw = decoder1->reader->jpeg_header_raw;
    void* jpeg_header_croped = onlineROI(jpeg_header_raw, 100, 100, 128, 128);
    if (get_jpeg_header_status(jpeg_header_croped) == 1) {
        warmup(input, decoder2, image, image_size, jpeg_header_croped, 1000);
    } else {
        warmup(input, decoder2, image, image_size, nullptr, 1000);
    }
    // return 0;
    assert(argc >= 6);
    int thread_num = atoi(argv[3]);
    int max_iter = atoi(argv[4]);
    int batch_size = atoi(argv[5]);
    printf("thread num: %d\n", thread_num);
    printf("iter: %d\n", max_iter);
    printf("batch size: %d\n", batch_size);

    vector<vector<uint8_t>> image_data;
    for (int i = 0; i < batch_size; i++) {
        image_data.emplace_back(image, image + image_size);
    }

    auto jpeg_header_crop = onlineROI(jpeg_header_raw, 0, 0, 224, 224);
    gpujpeg_decoder* decoder3 = gpujpeg_decoder_create(0);
    warmup(input, decoder3, image, image_size, jpeg_header_crop, 1000);

    decode(image_data, thread_num, max_iter, jpeg_header_crop);

    // Destroy image
    gpujpeg_image_destroy(image);
    // Destroy decoder
    gpujpeg_decoder_destroy(decoder1);
    return 0;
}