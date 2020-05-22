#include "../include/jpeg_decoder_export.h"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "jpeg_decoder.hpp"

using std::string;
using std::vector;

struct block_offset_s *unpack_jpeg_comment_section(char *data, size_t length, size_t *out_num_element, int *data_len) {
    jpeg_dec::RecoredFileds record = jpeg_dec::unpack_jpeg_comment_section(data, length, out_num_element);
    struct block_offset_s *ret = (struct block_offset_s *) malloc(sizeof(struct block_offset_s) * (*out_num_element));
    for (size_t i = 0; i < *out_num_element; i++) {
        ret[i].byte_offset = record.blockpos[i].first;
        ret[i].bit_offset = record.blockpos[i].second;
        ret[i].dc_value = record.dc_value[i];
    }
    *data_len = record.data_len;
    return ret;
}

int writeBMP(const char *filename, const unsigned char *chanR, const unsigned char *chanG, const unsigned char *chanB,
             int width, int height) {

    return jpeg_dec::writeBMP(filename, chanR, chanG, chanB, width, height);
}

void dumpFile(const char *filename, const char *content, size_t length) {
    std::ofstream of(filename, std::ofstream::binary | std::ofstream::trunc);
    of.write(content, length);
    of.close();
}

struct JPEG_HEADER {
    vector<vector<uint8_t>> dqt_table;
    vector<uint8_t> sof0;
    vector<vector<uint8_t>> dht;
    vector<uint8_t> sos_first_part;
    vector<uint8_t> sos_second_part;
    struct block_offset_s *block_offsets;
    int blocks_num;
    int block_offset_data_len;

    uint8_t status;  // 0->new created, 1->with data;
};

void *create_jpeg_header() {
    JPEG_HEADER *ret = new struct JPEG_HEADER;
    ret->status = 0;
    return static_cast<void *>(ret);
}

void destory_jpeg_header(void *jpeg_header_raw) {
    if (jpeg_header_raw) {
        JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
        delete jpeg_header;
    }
}

void set_jpeg_header_status(void *jpeg_header_raw, uint8_t status) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->status = status;
}
uint8_t get_jpeg_header_status(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->status;
}
void set_block_offsets(void *jpeg_header_raw, struct block_offset_s *block_offs, int length, int data_len) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->block_offsets = block_offs;
    jpeg_header->blocks_num = length;
    jpeg_header->block_offset_data_len = data_len;
}

struct block_offset_s *get_block_offsets(void *jpeg_header_raw, int *length, int *data_len) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    *length = jpeg_header->blocks_num;
    *data_len = jpeg_header->block_offset_data_len;
    return jpeg_header->block_offsets;
}

void set_dqt_table(void *jpeg_header_raw, int length, uint8_t *dqt_content) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->dqt_table.emplace_back(dqt_content, dqt_content + length);
}
uint8_t *get_dqt_table(void *jpeg_header_raw, int id) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->dqt_table[id].data();
}
int get_dqt_table_size(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->dqt_table.size();
}
void set_sof0(void *jpeg_header_raw, int length, uint8_t *sof0) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->sof0.resize(length);
    memcpy(jpeg_header->sof0.data(), sof0, length);
}
uint8_t *get_sof0_table(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->sof0.data();
}
void set_dht(void *jpeg_header_raw, int length, uint8_t *dht) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->dht.emplace_back(dht, dht + length);
}
int get_dht_table_size(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->dht.size();
}
uint8_t *get_dht_table(void *jpeg_header_raw, int id) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->dht[id].data();
}
void set_sos_1st(void *jpeg_header_raw, int length, uint8_t *sos_1st) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->sos_first_part.resize(length);
    memcpy(jpeg_header->sos_first_part.data(), sos_1st, length);
}
void set_sos_2nd(void *jpeg_header_raw, int length, uint8_t *sos_2nd) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->sos_second_part.resize(length);
    memcpy(jpeg_header->sos_second_part.data(), sos_2nd, length);
}

uint8_t *get_sos_1st(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->sos_first_part.data();
}
uint8_t *get_sos_2nd(void *jpeg_header_raw, int *length) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    *length = jpeg_header->sos_second_part.size();
    return jpeg_header->sos_second_part.data();
}

// char* onlineROI(const char* data, size_t length, size_t* out_length, const struct block_offset_s* block_off, int
// blocks,
//                 int ROI_w, int ROI_h, int ROI_x, int ROI_y) {
//     assert(block_off != nullptr);

//     *out_length = length;
//     return data;
// }