#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "jpeg_decoder_def.hpp"

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
const uint8_t DRI_SYM = 0xDD;   // restart_marker

const uint8_t POS_RECORD_SEG1 = 9;  // 9 bit per block offset(record a relative length)
const uint8_t POS_RECORD_SEG2 = 3;  // 3 bit to record which bit in a byte
struct RecoredFileds;

uint16_t big_endian_bytes2_uint(void *data);
void bytes2_big_endian_uint(uint16_t len, uint8_t *target_ptr);
int writeBMP(const char *filename, const unsigned char *chanR, const unsigned char *chanG, const unsigned char *chanB,
             int width, int height);
RecoredFileds unpack_jpeg_comment_section(char *data, size_t length, size_t *out_length);
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
    vector<vector<vector<float>>> mcu;
    uint16_t h_mcu_num;
    uint16_t w_mcu_num;
    MCUs() { mcu.resize(3); }
};
struct RGBPix {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};
struct RecoredFileds {
    bool scan_finish = false;
    size_t offset = 0;
    size_t total_blocks = 0;
    int data_len;
    vector<int16_t> dc_value;
    vector<std::pair<size_t, uint8_t>> blockpos;
    vector<uint8_t> blockpos_compact;
    // string str = "hello";
    template <class Archive>
    void serialize(Archive &archive) {
        archive(offset, total_blocks, blockpos_compact, dc_value, data_len);
    }
};

struct Image_struct {
    APPinfo app0;
    vector<vector<float>> dqt_tables;
    SOFInfo sof;
    HuffmanTable huffmanTable;
    vector<uint8_t> table_mapping_dc;
    vector<uint8_t> table_mapping_ac;
    MCUs mcus;
    RecoredFileds recordFileds;
    vector<int16_t> last_dc;
    vector<RGBPix> rgb;
    size_t sof0_offset = 0;
    int restart_interval = 0;
    Image_struct() {
        for (int i = 0; i < 4; i++) {
            last_dc.push_back(0.0);
        }
    }
};
class OBitStream {
   public:
    OBitStream() : total_bit_count(0) { new_byte(); }
    ~OBitStream() = default;
    void write_bit(uint8_t bit) {
        if (bit) {
            data[current_byte_offset] |= 1 << (7 - offset_in_byte);
        }
        offset_in_byte++;
        if (offset_in_byte == 8) {
            new_byte();
        }
        total_bit_count++;
    }

    vector<uint8_t> get_data() { return data; }
    size_t total_write_bit() { return total_bit_count; }

   private:
    uint8_t offset_in_byte;
    size_t current_byte_offset;
    size_t total_bit_count;
    vector<uint8_t> data;
    void new_byte() {
        data.push_back(0);
        current_byte_offset = data.size() - 1;
        offset_in_byte = 0;
    }
};

class BitStream {
   public:
    BitStream(uint8_t *ptr, bool jpeg_safe) : JPEG_SAFE(jpeg_safe), ptr(ptr), pos(0), base_ptr(ptr) {
        forward_a_byte();
    };
    BitStream(uint8_t *ptr, uint8_t *base_ptr) : ptr(ptr), pos(0), base_ptr(base_ptr) { forward_a_byte(); };
    BitStream(uint8_t *ptr, uint8_t bit_off, uint8_t *base_ptr) { init(ptr, bit_off, base_ptr); }
    void init(uint8_t *ptr_t, uint8_t bit_off_t, uint8_t *base_ptr_t) {
        this->ptr = ptr_t;
        this->pos = bit_off_t;
        this->base_ptr = base_ptr_t;
        this->cur_ptr = ptr_t;
        this->ptr++;
        if (JPEG_SAFE && this->cur_ptr[0] == 0xFF) {  // remember to skip 0x00 !
            assert(this->ptr[0] == 0x00);
            this->ptr++;
        }
    }
    uint8_t get_a_bit() {
        uint8_t ret = 0;
        if (cur_ptr[0] & (1 << (7 - pos))) {
            ret = 1;
        }
        pos++;
        if (pos == 8) {
            forward_a_byte();
        }
        return ret;
    }
    int16_t read_value(uint8_t len) { return read_value_int16(len); }
    int16_t read_value_int16(uint8_t len) {
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
        return ret;
    }

    void skip_value(uint8_t len) {
        for (uint8_t i = 0; i < len; i++) {
            get_a_bit();
        }
    }

    void skip_to_next_byte_with_rst() {
        if (pos != 0) {
            forward_a_byte();
        }
    }

    uint8_t *get_ptr() { return ptr; }
    size_t get_global_offset() { return cur_ptr - base_ptr; }
    uint8_t get_bit_offset() { return pos; }

   private:
    bool JPEG_SAFE = true;  // should we skip 0x00 after 0xFF?
    uint8_t *ptr;           // next
    uint8_t pos;
    uint8_t *cur_ptr;   // current ptr position
    uint8_t *base_ptr;  // the base ptr
    bool done = false;
    uint8_t current_rst_marker = 0xD0;

    void forward_a_byte() {
        assert(!done);
        cur_ptr = ptr;
        pos = 0;
        ptr++;
        if (JPEG_SAFE && cur_ptr[0] == 0xFF) {  // JPEG: 0xFF folows a 0x00
            if (ptr[0] == 0x00) {
                ptr++;
            } else if (ptr[0] == EOI_SYM) {
                done = true;
                return;
            } else if (ptr[0] <= 0xD7 && ptr[0] >= 0xD0) {
                if (ptr[0] != current_rst_marker) {
                    printf("error rst marker, expect: %x, found: %d\n", current_rst_marker, ptr[0]);
                }
                current_rst_marker++;
                if (current_rst_marker > 0xD7) {
                    current_rst_marker = 0xD0;
                }
                ptr++;
                forward_a_byte();  // forward a byte, (may be leading by 0xff)
            }
        }
    };
};

class JPEGDec {
   public:
    JPEGDec(const string &fname);
    JPEGDec(const vector<uint8_t> &image);
    JPEGDec(const uint8_t *image_raw, size_t len);
    ~JPEGDec();
    void Parser();
    size_t Parser_app0(uint8_t *data_ptr);
    size_t Parser_DQT(uint8_t *data_ptr);
    size_t Parser_SOF0(uint8_t *data_ptr);
    size_t Parser_DHT(uint8_t *data_ptr);
    size_t Parser_SOS(uint8_t *data_ptr);
    size_t Parser_MCUs(uint8_t *data_ptr);
    size_t Scan_MCUs(uint8_t *data_ptr);
    size_t Parser_DRI(uint8_t *data_ptr);
    void Decoding_on_BlockOffset();
    void compact_boundary();
    void WriteBoundarytoFile(const string &fname);

    void Dequantize();
    void ZigZag();
    void IDCT();
    void toRGB();
    void Dump(const string &fname);

    Image_struct get_imgstruct();

    JPEG_HEADER get_header();

   private:
    size_t parser_mcu(uint16_t h_mcu_idx, uint16_t w_mcu_idx);
    size_t read_block(uint8_t id, size_t block_idx);
    uint8_t match_huffman(std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> &map);
    Image_struct images;
    vector<uint8_t> data;
    std::unique_ptr<BitStream> bitStream;
    JPEG_HEADER header;
};

}  // namespace jpeg_dec