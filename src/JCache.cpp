#include "../include/JCache.hpp"
#include "../include/jpeg_decoder.hpp"
#include "../include/jpeg_decoder_export.h"
#include "spdlog/spdlog.h"
namespace jcache {
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

const uint8_t POS_RECORD_SEG1 = 9;  // 9 bit per block offset(record a relative length)
const uint8_t POS_RECORD_SEG2 = 3;  // 3 bit to record which bit in a byte

inline uint16_t big_endian_bytes2_uint(const void *data) {
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

inline void bytes2_big_endian_uint(uint16_t len, uint8_t *target_ptr) {
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

JCache::JCache(){};
bool JCache::putJPEG(const vector<uint8_t> &image, const string &filename) {
    jpeg_dec::JPEGDec dec(image);
    dec.Parser();
    map_[filename] = dec.get_header();
    return true;
}
bool JCache::putJPEG(const string &filename) {
    jpeg_dec::JPEGDec dec(filename);
    dec.Parser();
    map_[filename] = dec.get_header();
    return true;
}
JPEG_HEADER *JCache::getHeader(const string &filename) {
    auto e = map_.find(filename);
    if (e != map_.end()) {
        return &e->second;
    }
    return nullptr;
}
JPEG_HEADER *JCache::getHeaderwithCrop(const string &filename, int offset_x, int offset_y, int roi_width,
                                       int roi_height) {
    auto header = getHeader(filename);
    if (header == nullptr) {
        return nullptr;
    }
    return static_cast<JPEG_HEADER *>(onlineROI(header, offset_x, offset_y, roi_width, roi_height));
}
}  // namespace jcache