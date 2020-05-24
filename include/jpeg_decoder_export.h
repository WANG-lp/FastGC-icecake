/**
 * export c functions for my jpeg decoder
 *
 **/

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
};

struct JPEG_IMAGE {
    char* data;
    int data_length;
    struct block_offset_s* block_offs;
    int block_num;

    // JPEG header
    int width;
    int height;
    uint8_t sampling;
};

struct block_offset_s* unpack_jpeg_comment_section(char* data, size_t length, size_t* out_num_element, int* data_len);
int writeBMP(const char* filename, const unsigned char* chanR, const unsigned char* chanG, const unsigned char* chanB,
             int width, int height);
void* onlineROI(void* jpeg_header_raw, int offset_x, int offset_y, int roi_width, int roi_height);
void dumpFile(const char* filename, const char* content, size_t length);

void* create_jpeg_header();
void destory_jpeg_header(void* jpeg_header_raw);
void set_dqt_table(void* jpeg_header_raw, int length, uint8_t* dqt_content);
void set_sof0(void* jpeg_header_raw, int length, uint8_t* sof0);
void set_dht(void* jpeg_header_raw, int length, uint8_t* dht);
void set_sos_1st(void* jpeg_header_raw, int length, uint8_t* sos_1st);
void set_sos_2nd(void* jpeg_header_raw, int length, uint8_t* sos_2nd);
void set_block_offsets(void* jpeg_header_raw, struct block_offset_s* block_offs, int length, int data_len);
void set_jpeg_header_status(void* jpeg_header_raw, uint8_t status);
void set_jpeg_size(void* jpeg_header_raw, int width, int height);

int get_dqt_table_size(void* jpeg_header_raw);
uint8_t* get_dqt_table(void* jpeg_header_raw, int id);
uint8_t* get_sof0_table(void* jpeg_header_raw);
int get_dht_table_size(void* jpeg_header_raw);
uint8_t* get_dht_table(void* jpeg_header_raw, int id);
uint8_t* get_sos_1st(void* jpeg_header_raw);
uint8_t* get_sos_2nd(void* jpeg_header_raw, int* length);
struct block_offset_s* get_block_offsets(void* jpeg_header_raw, int* length, int* data_len);
uint8_t get_jpeg_header_status(void* jpeg_header_raw);

#ifdef __cplusplus
}
#endif

#endif  // JPEG_DEC_EXPORT_H
