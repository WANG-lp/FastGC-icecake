#pragma once
#include "jpeg_decoder_export.h"

#include <string>
#include <vector>

using std::string;
using std::vector;
#include "../GPUJPEG/src/export.h"

struct JPEG_HEADER {
    vector<vector<uint8_t>> dqt_table;
    vector<uint8_t> sof0;
    vector<vector<uint8_t>> dht;
    vector<uint8_t> sos_first_part;
    vector<uint8_t> sos_second_part;
    // |------16bit--------|-----13bit-----|---3bit---|
    //       dc_value            offset      bit_offset
    vector<uint32_t> blockpos_compact;
    vector<block_offset_s> blockpos;
    int blocks_num;

    int width;
    int height;

    uint8_t status = 0;  // 0->new created, 1->with data, 2->error;
};

struct JPEG_FAST_BINARY {
    vector<vector<uint16_t>> dqt_table;
    vector<gpujpeg_table_huffman_decoder> dht_table;
    int height, weidth;
    uint8_t comp_count;
    vector<uint8_t> sampling_factor_h;
    vector<uint8_t> sampling_factor_v;

    int blocks_num;
    vector<uint32_t> blockpos_compact;
    vector<block_offset_s> blockpos;

    vector<vector<uint8_t>> comp_dc_table_id;
    vector<vector<uint8_t>> comp_ac_table_id;

    int scan_segment_index;
    int scan_segment_count;
};
