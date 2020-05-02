#pragma once
#include <chrono>
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
std::chrono::steady_clock::time_point get_wall_time();
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
    std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> dc_tables[2];
    std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> ac_tables[2];
};

struct MCUs {
    vector<vector<float>> mcu[3];
    uint16_t h_mcu_num;
    uint16_t w_mcu_num;
};
struct RGBPix {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};
struct BlockPos {
    uint32_t dc_pos_byte;
    uint8_t dc_pos_bit;
    uint32_t ac_pos_byte;
    uint8_t ac_pos_bit;
};
struct Image_struct {
    APPinfo app0;
    vector<vector<float>> dqt_tables;
    SOFInfo sof;
    HuffmanTable huffmanTable;
    vector<uint8_t> table_mapping_dc;
    vector<uint8_t> table_mapping_ac;
    MCUs mcus;
    vector<BlockPos> blockpos;
    float last_dc[3];
    vector<RGBPix> rgb;

    uint8_t tmp_byte;
    uint8_t tmp_byte_consume_pos;
    uint8_t *global_data_reamins;
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

    void Dequantize(size_t idx);
    void ZigZag(size_t idx);
    void IDCT(size_t idx);
    void toRGB(size_t idx);
    void Dump(size_t idx, const string &fname);

    Image_struct get_imgstruct(size_t idx);

   private:
    uint8_t get_a_bit(size_t idx);
    float read_value(size_t idx, uint8_t code_len);
    void set_up_bit_stream(size_t idx, uint8_t *init_ptr);
    size_t parser_mcu(size_t idx, uint16_t h_mcu_idx, uint16_t w_mcu_idx);
    size_t read_block(size_t idx, uint8_t id, size_t block_idx);
    uint8_t match_huffman(std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> &map, size_t idx);
    vector<Image_struct> images;
    vector<vector<uint8_t>> data;
};

}  // namespace jpeg_dec