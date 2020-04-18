#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using std::string;
using std::vector;

namespace jpeg_dec {
const uint8_t DC0_SYM = 0x00;
const uint8_t DC1_SYM = 0x01;
const uint8_t AC0_SYM = 0x10;
const uint8_t AC1_SYM = 0x11;
const uint8_t MARKER_PREFIX = 0xFF;
const uint8_t SOI_SYM = 0xD8;   // start of image
const uint8_t EOI_SYM = 0xD9;   // end of image
const uint8_t APP0_SYM = 0xE0;  // JFIF info
const uint8_t DQT_SYM = 0xDB;   // DQT (define quantization table)
const uint8_t DHT_SYM = 0xC4;   // DHT (define huffman table)
const uint8_t SOF0_SYM = 0xC0;  // start of frame (baseline)
const uint8_t SOS_SYM = 0xDA;   // SOS, start of scan
const uint8_t COM_SYM = 0xFE;   // comment

uint16_t big_endian_bytes2_uint(void *data);

struct APPinfo {
    vector<uint8_t> identifier = vector<uint8_t>(5);
    uint8_t version_major;
    uint8_t version_minor;
    uint8_t units;
    uint16_t x_density;
    uint16_t y_density;
    uint8_t x_thumbnail;
    uint8_t y_thumbnail;
};

struct ComponentInfo {
    uint8_t horizontal_sampling;
    uint8_t vertical_sampling;
    uint8_t quant_table_id;
};

struct SOFInfo {
    uint8_t precision;
    uint16_t height;
    uint16_t width;
    uint8_t max_horizontal_sampling;
    uint8_t max_vertical_sampling;
    vector<ComponentInfo> component_infos;
};

struct HuffmanTable {
    vector<std::unordered_map<std::tuple<uint8_t, uint16_t>, uint16_t>> dc_tables;
    vector<std::unordered_map<std::tuple<uint8_t, uint16_t>, uint16_t>> ac_tables;
};

struct MCUs {
    vector<vector<uint8_t>> mcu[3];
    uint16_t h_mcu;
    uint16_t w_mcu;
};

struct Image_struct {
    APPinfo app0;
    vector<vector<float>> dqt_tables;
    SOFInfo sof;
    HuffmanTable huffmanTable;
    vector<uint8_t> table_mapping_dc;
    vector<uint8_t> table_mapping_ac;
    MCUs mcus;
};

class JPEGDec {
   public:
    JPEGDec(const string &fname);
    ~JPEGDec();
    void Parser(size_t idx);
    size_t Parser_app0(size_t idx, uint8_t *data_ptr);
    size_t Parser_DQT(size_t idx, uint8_t *data_ptr);
    size_t Parser_SOF0(size_t idx, uint8_t *data_ptr);
    size_t Parser_DHT(size_t idx, uint8_t *data_ptr);
    size_t Parser_SOS(size_t idx, uint8_t *data_ptr);
    size_t Parser_MCUs(size_t idx, uint8_t *data_ptr);

   private:
    vector<Image_struct> images;
    vector<vector<uint8_t>> data;
};

}  // namespace jpeg_dec