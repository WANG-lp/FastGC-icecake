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
    vector<block_offset_s> block_offsets;
    int blocks_num;

    int width;
    int height;

    uint8_t status;  // 0->new created, 1->with data;
};