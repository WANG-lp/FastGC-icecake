#include "../include/jpeg_decoder_export.h"
#include "jpeg_decoder.hpp"

#include <fstream>

struct block_offset_s* unpack_jpeg_comment_section(char* data, size_t length, size_t* out_num_element, int* data_len) {
    jpeg_dec::RecoredFileds record = jpeg_dec::unpack_jpeg_comment_section(data, length, out_num_element);
    struct block_offset_s* ret = (struct block_offset_s*) malloc(sizeof(struct block_offset_s) * (*out_num_element));
    for (size_t i = 0; i < *out_num_element; i++) {
        ret[i].byte_offset = record.blockpos[i].first;
        ret[i].bit_offset = record.blockpos[i].second;
        ret[i].dc_value = record.dc_value[i];
    }
    *data_len = record.data_len;
    return ret;
}

int writeBMP(const char* filename, const unsigned char* chanR, const unsigned char* chanG, const unsigned char* chanB,
             int width, int height) {

    return jpeg_dec::writeBMP(filename, chanR, chanG, chanB, width, height);
}

void dumpFile(const char* filename, const char* content, size_t length) {
    std::ofstream of(filename, std::ofstream::binary | std::ofstream::trunc);
    of.write(content, length);
    of.close();
}

char* onlineROI(const char* data, size_t length, size_t* out_length, const struct block_offset_s* block_off, int blocks,
                int ROI_w, int ROI_h, int ROI_x, int ROI_y) {
    assert(block_off != nullptr);

    *out_length = length;
    return data;
}