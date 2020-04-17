#include "../include/jpeg_decoder.hpp"
#include <spdlog/spdlog.h>

namespace jpeg_dec {

void print_app0(APPinfo &app0) {
    spdlog::info("App0:");
    spdlog::info("\tmajor_version: {}, minor_version: {}", app0.version_major, app0.version_minor);
    spdlog::info("\tidentifier: {} {} {} {} {}", app0.identifier[0], app0.identifier[1], app0.identifier[2],
                 app0.identifier[3], app0.identifier[4]);
    spdlog::info("\tunits: {}, x_density: {}, y_density: {}, x_thumbnail: {}, y_thumbnail: {}", app0.units,
                 app0.x_density, app0.y_density, app0.x_thumbnail, app0.y_thumbnail);
}
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
        if (c != MARKER_PREFIX) {
            continue;
        }
        off++;  // forward one byte
        c = data[idx][off];

        switch (c) {
            case 0x00: {
                // 0xFF00 does not represent any header marker
                break;
            };
            case SOI_SYM: {
                spdlog::info("start of image, offset: {}", off);
                break;
            };
            case EOI_SYM: {
                spdlog::info("end of image, offset: {}", off);
                return;
            };
            case APP0_SYM: {
                spdlog::info("APP0, JFIF mark, offset: {}", off);
                Parser_app0(idx, data[idx].data() + off + 1);
                print_app0(images[idx].app0);
                break;
            };
            case DQT_SYM: {
                spdlog::info("DQT, offset: {}", off);
                Parser_DQT(idx, data[idx].data() + off + 1);
                break;
            };
            case DHT_SYM: {
                spdlog::info("DHT, offset: {}", off);
                Parser_DHT(idx, data[idx].data() + off + 1);
                break;
            };
            case SOF0_SYM: {
                spdlog::info("SOF0(baseline), offset: {}", off);
                Parser_SOF0(idx, data[idx].data() + off + 1);
                break;
            };
            case SOS_SYM: {
                spdlog::info("SOS, begin data, offset: {}", off);
                break;
            }
            default: {
                spdlog::info("Unknown marker, offset: {}, byte: {}", off, c);
            }
        }
    }
}

size_t JPEGDec::Parser_app0(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    if (images.size() <= idx) {
        images.resize(idx + 1);
    }
    data_raw += 2;
    memcpy(images[idx].app0.identifier.data(), data_raw, 5);
    data_raw += 5;
    images[idx].app0.version_major = data_raw[0];
    data_raw += 1;
    images[idx].app0.version_minor = data_raw[0];
    data_raw += 1;
    images[idx].app0.units = data_raw[0];
    data_raw += 1;
    images[idx].app0.x_density = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images[idx].app0.y_density = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images[idx].app0.x_thumbnail = data_raw[0];
    data_raw += 1;
    images[idx].app0.y_thumbnail = data_raw[0];
    data_raw += 1;
    return len;
}
size_t JPEGDec::Parser_DQT(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_ptr);
    spdlog::info("DQT length: {}", len);
    len -= 2;
    data_raw += 2;
    while (len > 0) {
        uint8_t c = data_raw[0];
        uint8_t id = c & 0x0F;
        uint8_t precision = c >> 4;
        data_raw += 1;
        images[idx].dqt_tables.resize(id + 1);
        images[idx].dqt_tables[id].resize(64);
        float *table = images[idx].dqt_tables[id].data();
        if (precision == 0) {
            for (int i = 0; i < 64; i++) {
                table[i] = data_raw[i];
            }
            data_raw += 64;
            len -= 65;
        } else if (precision == 1) {
            for (int i = 0; i < 64; i++) {
                table[i] = big_endian_bytes2_uint(data_raw + i * 2);
            }
            data_raw += 128;
            len -= 129;
        } else {
            spdlog::error("Unknown DQT table");
            exit(1);
        }
        printf("DQT table %d, precision %d\n", id, precision);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                printf("%.2f ", table[i * 8 + j]);
            }
            printf("\n");
        }
    }
    return len;
}

size_t JPEGDec::Parser_SOF0(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    spdlog::info("SOF0 length: {}", len);
    images[idx].sof.precision = data_raw[0];
    data_raw += 1;
    images[idx].sof.height = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images[idx].sof.width = big_endian_bytes2_uint(data_raw);
    data_raw += 2;

    uint8_t num_component = data_raw[0];
    data_raw += 1;

    printf("precision: %d, height: %d, width: %d\n", images[idx].sof.precision, images[idx].sof.height,
           images[idx].sof.width);

    images[idx].sof.max_horizontal_sampling = 0;
    images[idx].sof.max_vertical_sampling = 0;
    for (uint8_t i = 0; i < num_component; i++) {
        uint8_t component_id = data_raw[i];
        data_raw += 1;
        ComponentInfo cinfo;
        uint8_t c = data_raw[0];
        data_raw += 1;
        cinfo.horizontal_sampling = c >> 4;
        cinfo.vertical_sampling = c & 0x0F;
        cinfo.quant_table_id = data_raw[0];
        data_raw += 1;
        if (cinfo.horizontal_sampling > images[idx].sof.max_horizontal_sampling) {
            images[idx].sof.max_horizontal_sampling = cinfo.horizontal_sampling;
        }
        if (cinfo.vertical_sampling > images[idx].sof.max_vertical_sampling) {
            images[idx].sof.max_vertical_sampling = cinfo.vertical_sampling;
        }
        images[idx].sof.component_infos.push_back(cinfo);
        printf("horizontal_sampling: %d, vertical_sampling: %d, quant_table_id: %d\n", cinfo.horizontal_sampling,
               cinfo.vertical_sampling, cinfo.quant_table_id);
        // printf("");
    }

    spdlog::info("max horizontal sampling: {}, max vertical sampling: {}", images[idx].sof.max_horizontal_sampling,
                 images[idx].sof.max_vertical_sampling);

    return len;
}

size_t JPEGDec::Parser_DHT(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    spdlog::info("DHT length: {}", len);
    return len;
}

}  // namespace jpeg_dec

int main(int argc, char **argv) {
    jpeg_dec::JPEGDec jpeg_dec(argv[1]);
    jpeg_dec.Header(0);
    return 0;
}
