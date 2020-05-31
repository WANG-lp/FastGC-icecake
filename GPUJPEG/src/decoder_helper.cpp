#include "decoder_helper.h"

#include <cassert>
#include <cstdio>

int init_decoder_with_image(struct gpujpeg_decoder* decoder, const char* image_path) {
    assert(decoder != NULL);
    int image_size = 0;
    uint8_t* image = NULL;
    if (gpujpeg_image_load_from_file(image_path, &image, &image_size) != 0) {
        fprintf(stderr, "Failed to load image [%s]!\n", image_path);
        return -1;
    }
    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
    int rc;
    if ((rc = gpujpeg_decoder_decode_phase1(decoder, image, image_size, NULL)) != 0) {
        fprintf(stderr, "Failed to decode image [%s]!\n", image_path);
        return -1;
    }

    decoder->max_height = decoder->reader->param_image.height;
    decoder->max_width = decoder->reader->param_image.width;
    decoder->max_comp = decoder->reader->param_image.comp_count;
    printf("%d %d %d\n", decoder->max_height, decoder->max_width, decoder->max_comp);
}