#include "../include/jpeg_decoder.hpp"
#include <spdlog/spdlog.h>
#include "hash_tuple.hpp"

namespace jpeg_dec {
// assumes little endian
void printBits(size_t const size, void const *const ptr) {
    unsigned char *b = (unsigned char *) ptr;
    unsigned char byte;
    int i, j;

    for (i = size - 1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

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

void JPEGDec::Parser(size_t idx) {
    for (size_t off = 0; off < data[idx].size();) {
        uint8_t c = data[idx][off];
        if (c != MARKER_PREFIX) {
            off += 1;
            continue;
        }
        off++;  // forward one byte
        c = data[idx][off];

        switch (c) {
            case 0x00: {
                off += 1;
                // 0xFF00 does not represent any header marker
                break;
            };
            case SOI_SYM: {
                spdlog::info("start of image, offset: {}", off);
                off += 1;
                break;
            };
            case EOI_SYM: {
                spdlog::info("end of image, offset: {}", off);
                off += 1;
                return;
            };
            case APP0_SYM: {
                spdlog::info("APP0, JFIF mark, offset: {}", off);
                off += Parser_app0(idx, data[idx].data() + off + 1);
                print_app0(images[idx].app0);
                break;
            };
            case DQT_SYM: {
                spdlog::info("DQT, offset: {}", off);
                off += Parser_DQT(idx, data[idx].data() + off + 1);
                break;
            };
            case DHT_SYM: {
                spdlog::info("DHT, offset: {}", off);
                off += Parser_DHT(idx, data[idx].data() + off + 1);
                break;
            };
            case SOF0_SYM: {
                spdlog::info("SOF0(baseline), offset: {}", off);
                off += Parser_SOF0(idx, data[idx].data() + off + 1);
                break;
            };
            case SOS_SYM: {
                spdlog::info("SOS, begin data, offset: {}", off);
                off += Parser_SOS(idx, data[idx].data() + off + 1);
                off += Parser_MCUs(idx, data[idx].data() + off);
                break;
            }
            default: {
                off += 1;
                spdlog::info("Unknown marker, offset: {}, byte: {}", off, c);
            }
        }
    }
}

size_t JPEGDec::Parser_app0(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
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
    return ret;
}
size_t JPEGDec::Parser_DQT(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_ptr);
    auto ret = len;
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
    return ret;
}

size_t JPEGDec::Parser_SOF0(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
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

    return ret;
}

size_t JPEGDec::Parser_DHT(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
    data_raw += 2;
    spdlog::info("DHT length: {}", len);
    len -= 2;
    while (len > 0) {
        uint8_t c = data_raw[0];
        data_raw += 1;
        uint8_t ac_dc = c >> 4;
        uint8_t id = c & 0x0F;
        vector<uint8_t> height_info(16);
        memcpy(height_info.data(), data_raw, 16);
        data_raw += 16;
        for (int i = 0; i < 16; i++) {
            printf("%d ", height_info[i]);
        }
        printf("\n");
        len -= 17;
        std::unordered_map<std::tuple<uint8_t, uint16_t>, uint16_t> map;
        uint16_t code = 0;
        for (uint8_t h = 0; h < 16; h++) {
            for (int i = 0; i < height_info[h]; i++) {
                uint8_t symb = data_raw[0];
                data_raw += 1;
                map.emplace(std::tuple<uint8_t, uint16_t>(h + 1, code), symb);
                // printBits(2, &code);
                // printf(" %#04X\n", symb);
                code += 1;
                len -= 1;
            }
            code <<= 1;
        }
        if (ac_dc == 0) {  // dc
            images[idx].huffmanTable.dc_tables.resize(id + 1);
            images[idx].huffmanTable.dc_tables[id] = map;
        } else {  // ac
            images[idx].huffmanTable.ac_tables.resize(id + 1);
            images[idx].huffmanTable.ac_tables[id] = map;
        }
    }
    return ret;
}

size_t JPEGDec::Parser_SOS(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
    data_raw += 2;
    spdlog::info("SOS length: {}", len);

    auto &table_mapping_dc = images[idx].table_mapping_dc;
    auto &table_mapping_ac = images[idx].table_mapping_ac;
    table_mapping_dc.resize(3);
    table_mapping_ac.resize(3);

    uint8_t component_num = data_raw[0];
    data_raw += 1;

    // we only support 3 channel (color) images
    assert(component_num == 3);

    for (uint8_t i = 0; i < 3; i++) {
        uint8_t comp = data_raw[0];
        uint8_t id = data_raw[1];
        data_raw += 2;
        uint8_t dc_id = (id >> 4);
        uint8_t ac_id = (id & 0x0F);
        table_mapping_dc[comp - 1] = dc_id;
        table_mapping_ac[comp - 1] = ac_id;
        printf("%d components, dc ht id: %d, ac ht id: %d\n", comp, dc_id, ac_id);
    }

    // following is fixed in JPEG-baseline
    assert(data_raw[0] == 0x00);
    assert(data_raw[1] == 0x3F);
    assert(data_raw[2] == 0x00);

    data_raw += 3;

    return ret;
}

size_t parser_mcu(Image_struct &img, uint8_t *data_ptr) {
    for (uint8_t id = 0; id < 3; id++) {
        uint16_t height = img.sof.component_infos[id].vertical_sampling;
        uint16_t width = img.sof.component_infos[id].horizontal_sampling;
        auto &dc_table = img.huffmanTable.dc_tables[img.table_mapping_dc[id]];
        auto &ac_table = img.huffmanTable.ac_tables[img.table_mapping_ac[id]];
        for (uint16_t h = 0; h < height; h++) {
            for (uint16_t w = 0; w < width; w++) {
                // img.mcus.mcu[id]
            }
        }
    }
    return 0;
}

size_t JPEGDec::Parser_MCUs(size_t idx, uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto sof_info = images[idx].sof;
    auto image_width = sof_info.width;
    auto image_height = sof_info.height;

    uint16_t w = (image_width - 1) / (8 * sof_info.max_horizontal_sampling) + 1;
    uint16_t h = (image_height - 1) / (8 * sof_info.max_vertical_sampling) + 1;
    spdlog::info("width has {} MCUs, height has {} MCUs", w, h);
    images[idx].mcus.w_mcu = w;
    images[idx].mcus.h_mcu = h;
    images[idx].mcus.mcu[0].resize(w * h);
    images[idx].mcus.mcu[1].resize(w * h);
    images[idx].mcus.mcu[2].resize(w * h);
    for (uint16_t i = 0; i < h; i++) {
        for (uint16_t j = 0; j < w; j++) {
            data_raw += parser_mcu(images[idx], data_raw);
        }
    }
    return 1;
}

}  // namespace jpeg_dec

int main(int argc, char **argv) {
    jpeg_dec::JPEGDec jpeg_dec(argv[1]);
    jpeg_dec.Parser(0);
    return 0;
}
