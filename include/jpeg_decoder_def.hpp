#pragma once
#include "jpeg_decoder_export.h"

#include <string>
#include <vector>

using std::string;
using std::vector;

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