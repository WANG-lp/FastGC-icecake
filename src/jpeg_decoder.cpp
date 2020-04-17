#include "../include/jpeg_decoder.hpp"
#include <spdlog/spdlog.h>

namespace jpeg_dec {
uint16_t big_endian_bytes2_uint(void *data) {
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

JPEGDec::JPEGDec(const string &fname) {
    std::ifstream ifs(fname, std::ios::binary | std::ios::ate);
    if (!ifs.good()) {
        spdlog::critical("cannot open file {}", fname);
        exit(1);
    }
    std::streamsize fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(1);
    data[0].resize(fsize);
    ifs.read((char *) data[0].data(), fsize);
}
JPEGDec::~JPEGDec() {}

void JPEGDec::Header(size_t idx) {
    for (size_t off = 0; off < data[idx].size(); off++) {
        uint8_t c = data[idx][off];
        if (c != 0xFF) {
            continue;
        }
        off++;  // forward one byte
        c = data[idx][off];

        switch (c) {
            case 0x00: {
                // 0xFF00 does not represent any header marker
                break;
            };
            case 0xD8: {
                spdlog::info("start of image, offset: {}", off);
                break;
            };
            case 0xD9: {
                spdlog::info("end of image, offset: {}", off);
                return;
            };
            case 0xE0: {
                spdlog::info("APP0, JFIF mark, offset: {}", off);
                break;
            };
            case 0xDB: {
                spdlog::info("DQT, offset: {}", off);
                break;
            };
            case 0xC4: {
                spdlog::info("DHT, offset: {}", off);
                break;
            };
            case 0xC0: {
                spdlog::info("SOF0(baseline), offset: {}", off);
                break;
            };
            case 0xDA: {
                spdlog::info("SOS, begin data, offset: {}", off);
                break;
            }
            default: {
                spdlog::info("Unknown marker, offset: {}, byte: {}", off, c);
            }
        }
    }
}

}  // namespace jpeg_dec

int main(int argc, char **argv) {
    jpeg_dec::JPEGDec jpeg_dec(argv[1]);
    jpeg_dec.Header(0);
    return 0;
}
