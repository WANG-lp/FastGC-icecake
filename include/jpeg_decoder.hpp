#pragma once
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
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
struct Image_struct {
    APPinfo app0;
    vector<vector<float>> dqt_tables;
    SOFInfo sof;
    HuffmanTable huffmanTable;
    vector<uint8_t> table_mapping_dc;
    vector<uint8_t> table_mapping_ac;
    MCUs mcus;
    vector<uint32_t> blockpos;  // high-29bit for <=512MB offset, low-3bit for 8 bit position in a byte
    float last_dc[3];
    vector<RGBPix> rgb;
};

class BitStream {
   public:
    BitStream(uint8_t *ptr, size_t global_off) : ptr(ptr), pos(0), global_off(global_off) { forward_a_byte(); };
    uint8_t get_a_bit() {
        uint8_t ret = 0;
        if (tmp_byte & (1 << (7 - pos))) {
            ret = 1;
        }
        pos++;
        if (pos == 8) {
            forward_a_byte();
        }
        return ret;
    }
    float read_value(uint8_t len) {
        int16_t ret = 1;
        uint8_t first_bit = get_a_bit();
        for (uint8_t i = 1; i < len; i++) {
            uint8_t bit = get_a_bit();
            ret = ret << 1;
            if (first_bit == bit) {
                ret += 1;
            } else {
                ret += 0;
            }
        }
        if (first_bit == 0) {
            ret = -ret;
        }
        return (float) ret;
    }

    void skip_value(uint8_t len) {
        for (uint8_t i = 0; i < len; i++) {
            get_a_bit();
        }
    }

    uint8_t *get_ptr() { return ptr; }
    size_t get_global_offset() { return global_off; }
    uint8_t get_bit_offset() { return pos; }

   private:
    uint8_t tmp_byte;
    uint8_t *ptr;
    uint8_t pos;
    size_t global_off;

    void forward_a_byte() {
        tmp_byte = ptr[0];
        pos = 0;
        ptr++;
        global_off++;
        if (tmp_byte == 0xFF) {  // JPEG: 0xFF folows a 0x00
            assert(ptr[0] == 0x00);
            ptr++;
            global_off++;
        }
    };
};

class JPEGDec {
   public:
    JPEGDec(const string &fname);
    ~JPEGDec();
    void Parser();
    size_t Parser_app0(uint8_t *data_ptr);
    size_t Parser_DQT(uint8_t *data_ptr);
    size_t Parser_SOF0(uint8_t *data_ptr);
    size_t Parser_DHT(uint8_t *data_ptr);
    size_t Parser_SOS(uint8_t *data_ptr);
    size_t Parser_MCUs(uint8_t *data_ptr);
    size_t Scan_MCUs(uint8_t *data_ptr);
    void Decoding_on_BlockOffset();

    void Dequantize();
    void ZigZag();
    void IDCT();
    void toRGB();
    void Dump(const string &fname);

    Image_struct get_imgstruct();

   private:
    size_t parser_mcu(uint16_t h_mcu_idx, uint16_t w_mcu_idx);
    size_t read_block(uint8_t id, size_t block_idx);
    uint8_t match_huffman(std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> &map);
    Image_struct images;
    vector<uint8_t> data;
    std::unique_ptr<BitStream> bitStream;
    std::ofstream logf;
};

}  // namespace jpeg_dec