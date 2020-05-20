/**
 * export c functions for my jpeg decoder
 * */

#ifndef JPEG_DEC_EXPORT_H
#define JPEG_DEC_EXPORT_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct block_offset_s {
    int byte_offset;
    unsigned char bit_offset;
    int16_t dc_value;
    int data_len;
};

struct block_offset_s* unpack_jpeg_comment_section(char* data, size_t length, size_t* out_num_element, int* data_len);
int writeBMP(const char* filename, const unsigned char* chanR, const unsigned char* chanG, const unsigned char* chanB,
             int width, int height);
char* onlineROI(char* data, size_t length, size_t* out_length, const struct block_offset_s* block_off, int blocks,
                int ROI_w, int ROI_h, int ROI_x, int ROI_y);
void dumpFile(const char* filename, const char* content, size_t length);
#ifdef __cplusplus
}
#endif

#endif  // JPEG_DEC_EXPORT_H
