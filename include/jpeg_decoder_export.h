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

struct block_offset_s {
    size_t byte_offset;
    unsigned char bit_offset;
};

block_offset_s* unpack_jpeg_comment_section(char* data, size_t length, size_t* out_num_element);

#ifdef __cplusplus
}
#endif

#endif  // JPEG_DEC_EXPORT_H
