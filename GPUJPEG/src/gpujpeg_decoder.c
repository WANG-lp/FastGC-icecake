/**
 * @file
 * Copyright (c) 2011-2019, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <assert.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <string.h>

#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_util.h"

/* Documented at declaration */
void gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output) {

    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer) {
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
    output->data = custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture) {
    output->type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    output->data = NULL;
    output->data_size = 0;
    output->texture = texture;
}

/* Documented at declaration */
void gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output) {
    output->type = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void gpujpeg_decoder_output_set_custom_cuda(struct gpujpeg_decoder_output* output, uint8_t* d_custom_buffer) {
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    output->data = d_custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}
/* Documented at declaration */
struct gpujpeg_decoder* gpujpeg_decoder_create(cudaStream_t* stream) {
    struct gpujpeg_decoder* decoder = (struct gpujpeg_decoder*) malloc(sizeof(struct gpujpeg_decoder));
    if (decoder == NULL)
        return NULL;

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    decoder->coder_inited = false;
    // Set parameters
    memset(decoder, 0, sizeof(struct gpujpeg_decoder));
    gpujpeg_set_default_parameters(&coder->param);
    gpujpeg_image_set_default_parameters(&coder->param_image);
    coder->param_image.comp_count = 0;
    coder->param_image.width = 0;
    coder->param_image.height = 0;
    coder->param.restart_interval = 0;

    decoder->max_height = -1;
    decoder->max_width = -1;
    decoder->max_comp = -1;

    int result = 1;

    // Create reader
    decoder->reader = gpujpeg_reader_create();
    if (decoder->reader == NULL)
        result = 0;

    // Allocate quantization tables in device memory
    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        if (cudaSuccess != cudaMalloc((void**) &decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)))
            result = 0;
    }
    // Allocate huffman tables in device memory
    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        for (int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++) {
            if (cudaSuccess != cudaMalloc((void**) &decoder->d_table_huffman[comp_type][huff_type],
                                          sizeof(struct gpujpeg_table_huffman_decoder)))
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation", return NULL);

    // Init huffman encoder
    if (gpujpeg_huffman_gpu_decoder_init() != 0) {
        result = 0;
    }

    // Stream
    decoder->stream = stream;
    if (decoder->stream == NULL) {
        decoder->allocatedStream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
        if (cudaSuccess != cudaStreamCreate(decoder->allocatedStream)) {
            result = 0;
        }
        decoder->stream = decoder->allocatedStream;
    }

    if (result == 0) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }

    return decoder;
}

GPUJPEG_API struct gpujpeg_decoder* gpujpeg_decoder_create_with_max_image_size(cudaStream_t* stream,
                                                                               uint8_t* image_data,
                                                                               size_t image_data_len, void* jpeg_header,
                                                                               void* fast_bin) {
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(stream);
    if (decoder == NULL) {
        return NULL;
    }
    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

    // TODO: replace init with a dummy param_image
    int rc;
    rc = gpujpeg_decoder_decode_phase1(decoder, image_data, image_data_len, jpeg_header, fast_bin);
    if (rc) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }
    rc = gpujpeg_decoder_decode_phase2(decoder, NULL);
    if (rc) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }

    return decoder;
}

/* Documented at declaration */
int gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, struct gpujpeg_parameters* param,
                         struct gpujpeg_image_parameters* param_image) {
    assert(param_image->comp_count == 1 || param_image->comp_count == 3);

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Check if (re)inialization is needed
    int change = 0;
    change |= coder->param_image.width != param_image->width;
    change |= coder->param_image.height != param_image->height;
    change |= coder->param_image.comp_count != param_image->comp_count;
    change |= coder->param.restart_interval != param->restart_interval;
    change |= coder->param.interleaved != param->interleaved;
    change |= coder->param.color_space_internal != param->color_space_internal;
    for (int comp = 0; comp < param_image->comp_count; comp++) {
        change |= coder->param.sampling_factor[comp].horizontal != param->sampling_factor[comp].horizontal;
        change |= coder->param.sampling_factor[comp].vertical != param->sampling_factor[comp].vertical;
    }
    if (change == 0)
        return 0;

    // For now we can't reinitialize decoder, we can only do first initialization
    // if (coder->param_image.width != 0 || coder->param_image.height != 0 || coder->param_image.comp_count != 0) {
    //     fprintf(stderr, "[GPUJPEG] [Error] Can't reinitialize decoder, implement if needed!\n");
    //     return -1;
    // }
    // printf("%d %d %d\n", param_image->height, param_image->width, param_image->comp_count);
    // printf("%d %d %d\n", coder->param_image.height, coder->param_image.width, coder->param_image.comp_count);

    // if (param_image->height <= coder->param_image.height && param_image->width <= coder->param_image.width &&
    //     param_image->comp_count <= coder->param_image.comp_count) {
    //     coder->param_image.height = param_image->height;
    //     coder->param_image.width = param_image->width;
    //     coder->data_raw_size = gpujpeg_image_calculate_size(&coder->param_image);
    //     return 0;
    // }
    // Initialize coder
    bool early_exit = false;
    if (!decoder->coder_inited) {
        if (gpujpeg_coder_init(coder) != 0) {
            return -1;
        }
        decoder->coder_inited = true;
        decoder->max_comp = param_image->comp_count;
        decoder->max_height = param_image->height;
        decoder->max_width = param_image->width;

        // printf("max: %d %d %d\n", decoder->max_height, decoder->max_width, decoder->max_comp);

    } else {
        if (param_image->height > decoder->max_height || param_image->width > decoder->max_width ||
            param_image->comp_count > decoder->max_comp) {
            fprintf(stderr, "[GPUJPEG] [Error] Image is too large!\n");
            return -1;
        }
        early_exit = true;
    }

    if (0 == gpujpeg_coder_init_image(coder, param, param_image, decoder->stream)) {
        return -1;
    }
    if (early_exit) {
        return 0;
    }
    // Init postprocessor
    if (gpujpeg_preprocessor_decoder_init(&decoder->coder) != 0) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init postprocessor!\n");
        return -1;
    }

    return 0;
}

int gpujpeg_decoder_decode_phase1(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, void* jpeg_header,
                                  void* fast_bin) {
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    int rc;
    int unsupp_gpu_huffman_params = 0;

    assert(jpeg_header != NULL || (image != NULL && image_size > 0) || fast_bin != NULL);

// Read JPEG image data
#ifdef TIMER
    int64_t t1 = get_wall_time();
#endif
    if (fast_bin) {
        if (0 != (rc = gimg_reader_read_image_with_fast_binary(decoder, fast_bin))) {
            fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
            return rc;
        }

    } else if (jpeg_header) {

        if (0 != (rc = gpujpeg_reader_read_image_with_header(decoder, jpeg_header))) {
            fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
            return rc;
        }

    } else {

        if (0 != (rc = gpujpeg_reader_read_image(decoder, image, image_size))) {
            fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
            return rc;
        }
    }
#ifdef TIMER
    int64_t t2 = get_wall_time();
    if (fast_bin) {
        printf("read with fast_bin: %lf ms\n", (t2 - t1) * 1.0 / 1000);

    } else if (jpeg_header) {
        printf("read with jpeg_header: %lf ms\n", (t2 - t1) * 1.0 / 1000);

    } else {
        printf("read: %lf ms\n", (t2 - t1) * 1.0 / 1000);
    }
#endif

    assert(decoder->reader->param_image.pixel_format == GPUJPEG_444_U8_P012);
    return 0;
}

int gpujpeg_decoder_decode_phase2(struct gpujpeg_decoder* decoder, struct gpujpeg_decoder_output* output) {
    bool set_output = true;
    if (output == NULL) {
        set_output = false;
    }
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    int rc;
    if (decoder->reader->block_offsets) {
        if (decoder->coder.block_offsets == NULL ||
            decoder->reader->block_count >
                decoder->coder.block_offset_allocated) {  // only allocate if we have more blocks
            if (decoder->coder.block_offsets) {
                cudaFreeHost(decoder->coder.block_offsets);
                cudaFree(decoder->coder.d_block_offsets);
            }
            cudaMallocHost((void**) &decoder->coder.block_offsets,
                           sizeof(struct block_offset_s) * decoder->reader->block_count);
            cudaMalloc((void**) &decoder->coder.d_block_offsets,
                       sizeof(struct block_offset_s) * decoder->reader->block_count);
            decoder->coder.block_offset_allocated = decoder->reader->block_count;
        }
        decoder->coder.block_pos_num = decoder->reader->block_count;

        memcpy(decoder->coder.block_offsets, decoder->reader->block_offsets,
               sizeof(struct block_offset_s) * decoder->reader->block_count);
        // free(decoder->reader->block_offsets);
    }

    // check if params is ok for GPU decoder
    for (int i = 0; i < decoder->coder.param_image.comp_count; ++i) {
        // packed_block_info_ptr holds only component type
        if (decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] !=
            decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC]) {
            fprintf(stderr,
                    "[GPUJPEG] [Warning] Using different table DC/AC indices (%d and %d) for component %d (ID %d)! "
                    "Using Huffman CPU decoder. Please report to GPUJPEG developers.\n",
                    decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC],
                    decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC], i, decoder->comp_id[i]);
            // unsupp_gpu_huffman_params = 1;
        }
        // only DC/AC tables 0 and 1 are processed gpujpeg_huffman_decoder_table_kernel()
        if (decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] > 1 ||
            decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC] > 1) {
            fprintf(stderr,
                    "[GPUJPEG] [Warning] Using Huffman tables (%d, %d) implies extended process! Using Huffman CPU "
                    "decoder. Please report to GPUJPEG developers.\n",
                    decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC],
                    decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC]);
            // unsupp_gpu_huffman_params = 1;
        }
    }

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if (coder->segment_count < 32 && decoder->reader->block_offsets == NULL) {
        // printf("using CPU-huffman decoding!!!\n");
        memset(coder->data_quantized, 0, sizeof(int16_t) * coder->data_size);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
        if (0 != gpujpeg_huffman_cpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return -1;
        }
        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_huffman_cpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Copy quantized data to device memory from cpu memory
        cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t),
                        cudaMemcpyHostToDevice, *(decoder->stream));

        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        // GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
        // Copy scan data to device memory
        cudaMemcpyAsync(coder->d_data_compressed, coder->data_compressed,
                        decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment),
                        cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // copy block positions to device memory
        if (decoder->reader->block_offsets) {
            cudaMemcpyAsync(coder->d_block_offsets, decoder->coder.block_offsets,
                            decoder->coder.block_pos_num * sizeof(struct block_offset_s), cudaMemcpyHostToDevice,
                            *(decoder->stream));

            gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);
        }
        // Zero output memory
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));

        // cudaStreamSynchronize(*(decoder->stream));
        // GPUJPEG_CUSTOM_TIMER_START(decoder->gpu_huff_dec);
        // Perform huffman decoding
#ifdef TIMER
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, *(decoder->stream));
#endif
        if (0 != gpujpeg_huffman_gpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }
#ifdef TIMER
        cudaEventRecord(stop, *(decoder->stream));
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU huffman decoding: %f ms\n", milliseconds);
#endif
        // cudaStreamSynchronize(*(decoder->stream));
        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->gpu_huff_dec);
        // coder->duration_in_gpu_huffman_dec = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->gpu_huff_dec);
    }

    // cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_START(decoder->gpu_idct);

    // Perform IDCT and dequantization (own CUDA implementation)
    if (0 != gpujpeg_idct_gpu(decoder)) {
        return -1;
    }
    // cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->gpu_idct);
    // coder->duration_idct = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->gpu_idct);
    // Create buffers if not already created
    if (coder->data_raw == NULL) {
        if (cudaSuccess != cudaMallocHost((void**) &coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }
    if (coder->d_data_raw_allocated == NULL) {
        if (cudaSuccess != cudaMalloc((void**) &coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }

    if (!set_output) {
        return 0;
    }

    // Select CUDA output buffer
    if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image should be directly decoded into custom CUDA buffer
        coder->d_data_raw = output->data;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE &&
               output->texture->texture_callback_attach_opengl == NULL) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Use OpenGL texture as decoding destination
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);
        coder->d_data_raw = d_data;

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else {
        // Use internal CUDA buffer as decoding destination
        coder->d_data_raw = coder->d_data_raw_allocated;
    }
    // printf("color space %d, pixel format: %d\n", decoder->coder.param_image.color_space,
    //    decoder->coder.param_image.pixel_format);
    // Preprocessing
    rc = gpujpeg_preprocessor_decode(&decoder->coder, *(decoder->stream));
    if (rc != GPUJPEG_NOERR) {
        return rc;
    }

    // Wait for async operations before copying from the device
    // GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    // coder->duration_waiting = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->in_gpu);
    // coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->in_gpu);

    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);

    // Set decompressed image
    if (output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Set output to internal buffer
        output->data = coder->data_raw;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        assert(output->data != NULL);

        // Copy decompressed image to host memory
        cudaMemcpy(output->data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE) {
        // If OpenGL texture wasn't mapped and used directly for decoding into it
        if (output->texture->texture_callback_attach_opengl != NULL) {
            GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Map OpenGL texture
            int data_size = 0;
            uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
            assert(data_size == coder->data_raw_size);

            // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            // coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

            GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Copy decompressed image to texture pixel buffer object device data
            cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

            // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        }

        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER) {
        // Copy decompressed image to texture pixel buffer object device data
        output->data = coder->d_data_raw;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image was already directly decoded into custom CUDA buffer
        output->data = coder->d_data_raw;
    } else {
        // Unknown output type
        assert(0);
    }

    return 0;
}
/* Documented at declaration */
int gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size,
                           struct gpujpeg_decoder_output* output) {
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    int rc;

    // Reset durations
    // coder->duration_huffman_cpu = 0.0;
    // coder->duration_memory_map = 0.0;
    // coder->duration_memory_unmap = 0.0;
    // coder->duration_stream = 0.0;
    // coder->duration_in_gpu = 0.0;
    // coder->duration_waiting = 0.0;
    // coder->duration_in_gpu_huffman_dec = 0.0;
    // coder->duration_idct = 0.0;

    GPUJPEG_CUSTOM_TIMER_START(decoder->def);

    // Read JPEG image data
    if (0 != (rc = gpujpeg_reader_read_image(decoder, image, image_size))) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return rc;
    }

    assert(decoder->reader->param_image.pixel_format == GPUJPEG_444_U8_P012);

    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    // coder->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

    // printf("segment_count: %d\n", coder->segment_count);

    // if (decoder->reader->block_offsets) {
    //     if (decoder->coder.block_offsets) {
    //         cudaFree(decoder->coder.block_offsets);
    //         cudaFree(decoder->coder.d_block_offsets);
    //     }
    //     if (decoder->coder.d_dc_values) {
    //         cudaFree(decoder->coder.d_dc_values);
    //     }
    //     cudaMallocHost((void**) &decoder->coder.block_offsets,
    //                    sizeof(struct block_offset_s) * decoder->reader->block_count);
    //     memcpy(decoder->coder.block_offsets, decoder->reader->block_offsets,
    //            sizeof(struct block_offset_s) * decoder->reader->block_count);

    //     decoder->coder.block_pos_num = decoder->reader->block_count;
    //     free(decoder->reader->block_offsets);

    //     cudaMalloc((void**) &decoder->coder.d_block_offsets,
    //                sizeof(struct block_offset_s) * decoder->reader->block_count);
    //     cudaMalloc((void**) &decoder->coder.d_dc_values, sizeof(int16_t) * decoder->reader->block_count);
    // }

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if (coder->segment_count < 32) {
        // memset(coder->data_quantized, 0, sizeof(int16_t) * coder->data_size);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
        if (0 != gpujpeg_huffman_cpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return -1;
        }
        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_huffman_cpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Copy quantized data to device memory from cpu memory
        cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t),
                        cudaMemcpyHostToDevice, *(decoder->stream));

        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);

        // Reset huffman output
        // cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));

        // Copy scan data to device memory
        cudaMemcpyAsync(coder->d_data_compressed, coder->data_compressed,
                        decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment),
                        cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // copy block positions to device memory
        // if (decoder->reader->block_offsets) {
        //     cudaMemcpyAsync(coder->d_block_offsets, decoder->coder.block_offsets,
        //                     decoder->coder.block_pos_num * sizeof(struct block_offset_s), cudaMemcpyHostToDevice,
        //                     *(decoder->stream));

        //     gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);
        // }
        // Zero output memory
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));

// cudaStreamSynchronize(*(decoder->stream));
// GPUJPEG_CUSTOM_TIMER_START(decoder->gpu_huff_dec);
// Perform huffman decoding
#if TIMER
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, *(decoder->stream));

#endif
        if (0 != gpujpeg_huffman_gpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }
#if TIMER
        cudaEventRecord(stop, *(decoder->stream));
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU huffman decoding: %f ms\n", milliseconds);
#endif
        // cudaStreamSynchronize(*(decoder->stream));
        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->gpu_huff_dec);
        // coder->duration_in_gpu_huffman_dec = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->gpu_huff_dec);
    }

    // cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_START(decoder->gpu_idct);

    // Perform IDCT and dequantization (own CUDA implementation)
    if (0 != gpujpeg_idct_gpu(decoder)) {
        return -1;
    }
    // cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->gpu_idct);
    // coder->duration_idct = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->gpu_idct);
    // Create buffers if not already created
    if (coder->data_raw == NULL) {
        if (cudaSuccess != cudaMallocHost((void**) &coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }
    if (coder->d_data_raw_allocated == NULL) {
        if (cudaSuccess != cudaMalloc((void**) &coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }

    // Select CUDA output buffer
    if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image should be directly decoded into custom CUDA buffer
        coder->d_data_raw = output->data;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE &&
               output->texture->texture_callback_attach_opengl == NULL) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Use OpenGL texture as decoding destination
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);
        coder->d_data_raw = d_data;

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else {
        // Use internal CUDA buffer as decoding destination
        coder->d_data_raw = coder->d_data_raw_allocated;
    }
    // printf("color space %d, pixel format: %d\n", decoder->coder.param_image.color_space,
    //    decoder->coder.param_image.pixel_format);
    // Preprocessing
    rc = gpujpeg_preprocessor_decode(&decoder->coder, *(decoder->stream));
    if (rc != GPUJPEG_NOERR) {
        return rc;
    }

    // Wait for async operations before copying from the device
    // GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    cudaStreamSynchronize(*(decoder->stream));
    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    // coder->duration_waiting = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

    // GPUJPEG_CUSTOM_TIMER_STOP(decoder->in_gpu);
    // coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->in_gpu);

    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);

    // Set decompressed image
    if (output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER) {
        // GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Set output to internal buffer
        output->data = coder->data_raw;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER) {
        // GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        assert(output->data != NULL);

        // Copy decompressed image to host memory
        cudaMemcpy(output->data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE) {
        // If OpenGL texture wasn't mapped and used directly for decoding into it
        if (output->texture->texture_callback_attach_opengl != NULL) {
            // GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Map OpenGL texture
            int data_size = 0;
            uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
            assert(data_size == coder->data_raw_size);

            // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            // coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

            GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Copy decompressed image to texture pixel buffer object device data
            cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

            // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            // coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        }

        // GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        // GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        // coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER) {
        // Copy decompressed image to texture pixel buffer object device data
        output->data = coder->d_data_raw;
    } else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image was already directly decoded into custom CUDA buffer
        output->data = coder->d_data_raw;
    } else {
        // Unknown output type
        assert(0);
    }

    return 0;
}

void gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder, enum gpujpeg_color_space color_space,
                                       enum gpujpeg_pixel_format sampling_factor) {
    decoder->coder.param_image.color_space = color_space;
    decoder->coder.param_image.pixel_format = sampling_factor;
}

/* Documented at declaration */
int gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder) {
    assert(decoder != NULL);

    if (0 != gpujpeg_coder_deinit(&decoder->coder)) {
        return -1;
    }

    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            cudaFree(decoder->table_quantization[comp_type].d_table);
        }
    }

    if (decoder->reader != NULL) {
        gpujpeg_reader_destroy(decoder->reader);
    }

    if (decoder->allocatedStream != NULL) {
        cudaStreamDestroy(*(decoder->allocatedStream));
        free(decoder->allocatedStream);
        decoder->allocatedStream = NULL;
        decoder->stream = NULL;
    }

    free(decoder);

    return 0;
}

/// @copydetails gpujpeg_reader_get_image_info
int gpujpeg_decoder_get_image_info(uint8_t* image, int image_size, struct gpujpeg_image_parameters* param_image,
                                   int* segment_count) {
    return gpujpeg_reader_get_image_info(image, image_size, param_image, segment_count);
}

/* vi: set expandtab sw=4 : */
