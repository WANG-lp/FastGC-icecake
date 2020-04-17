#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
using std::string;
using std::vector;

namespace jpeg_dec {
const uint8_t DC0_SYM = 0x00;
const uint8_t DC1_SYM = 0x01;
const uint8_t AC0_SYM = 0x10;
const uint8_t AC1_SYM = 0x11;
uint16_t big_endian_bytes2_uint(void *data);

class JPEGDec {
   public:
    JPEGDec(const string &fname);
    ~JPEGDec();
    void Header(size_t idx);

   private:
    vector<vector<uint8_t>> data;
};

}  // namespace jpeg_dec