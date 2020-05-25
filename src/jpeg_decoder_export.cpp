#include "../include/jpeg_decoder_export.h"
#include "../include/jpeg_decoder_def.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "../include/jpeg_decoder.hpp"

using std::string;
using std::vector;

static inline uint16_t big_endian_bytes2_uint(void *data) {
    auto bytes = (uint8_t *) data;
    uint16_t res;
#ifdef _BIG_ENDIAN
    return *((uint16_t *) bytes);
#else
    unsigned char *internal_buf = (unsigned char *) &res;
    internal_buf[0] = bytes[1];
    internal_buf[1] = bytes[0];
    return res;
#endif
}

static inline void bytes2_big_endian_uint(uint16_t len, uint8_t *target_ptr) {
    unsigned char *b = (unsigned char *) target_ptr;
    unsigned char *p = (unsigned char *) &len;
#ifdef _BIG_ENDIAN
    b[0] = p[0];
    b[1] = p[1];
#else
    b[0] = p[1];
    b[1] = p[0];
#endif
}
struct block_offset_s *unpack_jpeg_comment_section(char *data, size_t length, size_t *out_num_element) {
    jpeg_dec::RecoredFileds record = jpeg_dec::unpack_jpeg_comment_section(data, length, out_num_element);
    struct block_offset_s *ret = (struct block_offset_s *) malloc(sizeof(struct block_offset_s) * (*out_num_element));
    for (size_t i = 0; i < *out_num_element; i++) {
        ret[i].byte_offset = record.blockpos[i].first;
        ret[i].bit_offset = record.blockpos[i].second;
        ret[i].dc_value = record.dc_value[i];
    }
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

void *create_jpeg_header() {
    JPEG_HEADER *ret = new struct JPEG_HEADER;
    ret->status = 0;
    ret->width = ret->height = 0;
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
void set_block_offsets(void *jpeg_header_raw, const struct block_offset_s *block_offs, int length) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    // jpeg_header->block_offsets = vector<struct block_offset_s>(block_offs, block_offs + length);
    jpeg_header->block_offsets.resize(length);
    for (int i = 0; i < length; i++) {
        jpeg_header->block_offsets[i].byte_offset = block_offs[i].byte_offset;
        jpeg_header->block_offsets[i].bit_offset = block_offs[i].bit_offset;

        jpeg_header->block_offsets[i].dc_value = block_offs[i].dc_value;
    }
    jpeg_header->blocks_num = length;
}

struct block_offset_s *get_block_offsets(void *jpeg_header_raw, int *length) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    *length = jpeg_header->blocks_num;
    return jpeg_header->block_offsets.data();
}
void set_jpeg_size(void *jpeg_header_raw, int width, int height) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->width = width;
    jpeg_header->height = height;
}
void set_dqt_table(void *jpeg_header_raw, int length, const uint8_t *dqt_content) {
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
void set_sof0(void *jpeg_header_raw, int length, const uint8_t *sof0) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->sof0.resize(length);
    memcpy(jpeg_header->sof0.data(), sof0, length);
}
uint8_t *get_sof0_table(void *jpeg_header_raw) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    return jpeg_header->sof0.data();
}
void set_dht(void *jpeg_header_raw, int length, const uint8_t *dht) {
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
void set_sos_1st(void *jpeg_header_raw, int length, const uint8_t *sos_1st) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    jpeg_header->sos_first_part.resize(length);
    memcpy(jpeg_header->sos_first_part.data(), sos_1st, length);
}
void set_sos_2nd(void *jpeg_header_raw, int length, const uint8_t *sos_2nd) {
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

void *onlineROI(void *jpeg_header_raw, int offset_x, int offset_y, int roi_width, int roi_height) {
    JPEG_HEADER *jpeg_header = static_cast<JPEG_HEADER *>(jpeg_header_raw);
    assert(jpeg_header->block_offsets.size() > 0 && jpeg_header->status == 1);

    int width_mcu = (jpeg_header->width + 7) / 8;

    // TODO: only support 444 subsampling
    int mcu_h_start = offset_y / 8;
    int mcu_w_start = offset_x / 8;
    int pixel_h_start = mcu_h_start * 8;
    int pixel_w_start = mcu_w_start * 8;

    int pixel_h_end = (offset_y + roi_height);
    int pixel_w_end = (offset_x + roi_width);
    if (pixel_h_end > jpeg_header->height) {
        pixel_h_end = jpeg_header->height;
    }
    if (pixel_w_end > jpeg_header->width) {
        pixel_w_end = jpeg_header->width;
    }

    int mcu_h_end = (pixel_h_end + 7) / 8;
    int mcu_w_end = (pixel_w_end + 7) / 8;

    printf("pixel_h_start: %d, pixel_h_end: %d\n", pixel_h_start, pixel_h_end);
    printf("pixel_w_start: %d, pixel_w_end: %d\n", pixel_w_start, pixel_w_end);

    printf("mcu_h_start: %d, mcu_h_end: %d\n", mcu_h_start, mcu_h_end);
    printf("mcu_w_start: %d, mcu_w_end: %d\n", mcu_w_start, mcu_w_end);

    struct JPEG_HEADER *ret = static_cast<struct JPEG_HEADER *>(create_jpeg_header());
    // copy constant parts
    ret->dqt_table = jpeg_header->dqt_table;
    ret->dht = jpeg_header->dht;
    ret->sos_first_part = jpeg_header->sos_first_part;

    int total_blocks = 3 * (mcu_h_end - mcu_h_start) * (mcu_w_end - mcu_w_start);

    ret->block_offsets.resize(total_blocks);
    auto &block_pos_s = ret->block_offsets;
    // printf("block_pos_s size: %d\n", block_pos_s.size());

    int block_count = 0;
    int curr_byte_pos = 0;
    vector<uint8_t> sos2_data;

    for (int h = mcu_h_start; h < mcu_h_end; h++) {
        int start_mcu_id = h * width_mcu + mcu_w_start;
        int end_mcu_id = h * width_mcu + mcu_w_end;
        int start_block_id = start_mcu_id * 3;
        int end_block_id = end_mcu_id * 3;
        // printf("mcu: %d,%d\n", start_mcu_id, end_mcu_id);
        int start_byte_off = jpeg_header->block_offsets[start_block_id].byte_offset;
        int end_byte_off = (end_block_id) < jpeg_header->blocks_num
                               ? jpeg_header->block_offsets[end_block_id].byte_offset
                               : jpeg_header->sos_second_part.size();

        sos2_data.resize(curr_byte_pos + (end_byte_off - start_byte_off));
        memcpy(sos2_data.data() + curr_byte_pos, jpeg_header->sos_second_part.data() + start_byte_off,
               end_byte_off - start_byte_off);

        for (int block_id = start_block_id; block_id < end_block_id; block_id++) {
            int tmp_byte_off = curr_byte_pos + (jpeg_header->block_offsets[block_id].byte_offset - start_byte_off);
            block_pos_s[block_count] = {tmp_byte_off, jpeg_header->block_offsets[block_id].bit_offset,
                                        jpeg_header->block_offsets[block_id].dc_value};
            block_count++;
        }
        curr_byte_pos += end_byte_off - start_byte_off;
    }

    ret->blocks_num = block_count;
    ret->sos_second_part = sos2_data;

    // printf("total_blocks: %d\n", block_count);
    // assert(ret->blocks_num == jpeg_header->blocks_num);
    // change width/height value in sof0
    ret->sof0 = jpeg_header->sof0;
    bytes2_big_endian_uint(pixel_h_end - pixel_h_start, ret->sof0.data() + 3);
    bytes2_big_endian_uint(pixel_w_end - pixel_w_start, ret->sof0.data() + 5);
    int height = big_endian_bytes2_uint(ret->sof0.data() + 3);
    int width = big_endian_bytes2_uint(ret->sof0.data() + 5);
    printf("height: %d, width: %d\n", height, width);
    // for (int i = 0; i < total_blocks; i++) {
    //     int flag = 1;
    //     for (int c = 0; c < 1; c++) {
    //         uint8_t t1 = jpeg_header->sos_second_part[jpeg_header->block_offsets[i].byte_offset + c];
    //         uint8_t t2 = ret->sos_second_part[ret->block_offsets[i].byte_offset + c];
    //         if (t1 != t2) {
    //             printf("block %d, error, should: %x, got: %x\n", i, t1, t2);
    //             flag = 0;
    //             break;
    //         }
    //     }
    //     if (!flag) {
    //         break;
    //     }
    // }
    // copy left parts
    ret->status = 1;  // set status to ready

    return static_cast<void *>(ret);
}