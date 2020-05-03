#include "../include/jpeg_decoder.hpp"
#include <spdlog/spdlog.h>
#include <cmath>
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

std::chrono::steady_clock::time_point get_wall_time() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    return begin;
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
    data.resize(fsize);
    ifs.read((char *) data.data(), fsize);

    logf.open("/tmp/log.txt", std::ofstream::out | std::ofstream::trunc);
    logf << "==========" << fname << "===========" << std::endl;
}
JPEGDec::~JPEGDec() {}

void JPEGDec::Parser() {
    for (size_t off = 0; off < data.size();) {
        uint8_t c = data[off];
        if (c != MARKER_PREFIX) {
            off += 1;
            continue;
        }
        off++;  // forward one byte
        c = data[off];

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
                off += Parser_app0(data.data() + off + 1);
                print_app0(images.app0);
                break;
            };
            case DQT_SYM: {
                spdlog::info("DQT, offset: {}", off);
                off += Parser_DQT(data.data() + off + 1);
                break;
            };
            case DHT_SYM: {
                spdlog::info("DHT, offset: {}", off);
                off += Parser_DHT(data.data() + off + 1);
                break;
            };
            case SOF0_SYM: {
                spdlog::info("SOF0(baseline), offset: {}", off);
                off += Parser_SOF0(data.data() + off + 1);
                break;
            };
            case SOS_SYM: {
                spdlog::info("SOS, begin data, offset: {}", off);
                off += Parser_SOS(data.data() + off + 1);
                off++;
                auto t_s = get_wall_time();
                // off += Parser_MCUs(data.data() + off);
                off += Scan_MCUs(data.data() + off);
                spdlog::info("huffman decoding time: {}us",
                             std::chrono::duration_cast<std::chrono::microseconds>(get_wall_time() - t_s).count());

                break;
            }
            case COM_SYM: {
                spdlog::info("COM, begin data, offset: {}", off);
                off++;
                break;
            }
            default: {
                off += 1;
                spdlog::info("Unknown marker, offset: {}, byte: {}", off, c);
            }
        }
    }
}

size_t JPEGDec::Parser_app0(uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;

    data_raw += 2;
    memcpy(images.app0.identifier.data(), data_raw, 5);
    data_raw += 5;
    images.app0.version_major = data_raw[0];
    data_raw += 1;
    images.app0.version_minor = data_raw[0];
    data_raw += 1;
    images.app0.units = data_raw[0];
    data_raw += 1;
    images.app0.x_density = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images.app0.y_density = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images.app0.x_thumbnail = data_raw[0];
    data_raw += 1;
    images.app0.y_thumbnail = data_raw[0];
    data_raw += 1;
    return ret;
}
size_t JPEGDec::Parser_DQT(uint8_t *data_ptr) {
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
        images.dqt_tables.resize(id + 1);
        images.dqt_tables[id].resize(64);
        float *table = images.dqt_tables[id].data();
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

size_t JPEGDec::Parser_SOF0(uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
    data_raw += 2;
    spdlog::info("SOF0 length: {}", len);
    images.sof.precision = data_raw[0];
    data_raw += 1;
    images.sof.height = big_endian_bytes2_uint(data_raw);
    data_raw += 2;
    images.sof.width = big_endian_bytes2_uint(data_raw);
    data_raw += 2;

    uint8_t num_component = data_raw[0];
    data_raw += 1;

    printf("precision: %d, height: %d, width: %d\n", images.sof.precision, images.sof.height, images.sof.width);

    images.sof.max_horizontal_sampling = 0;
    images.sof.max_vertical_sampling = 0;
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
        if (cinfo.horizontal_sampling > images.sof.max_horizontal_sampling) {
            images.sof.max_horizontal_sampling = cinfo.horizontal_sampling;
        }
        if (cinfo.vertical_sampling > images.sof.max_vertical_sampling) {
            images.sof.max_vertical_sampling = cinfo.vertical_sampling;
        }
        images.sof.component_infos.push_back(cinfo);
        printf("horizontal_sampling: %d, vertical_sampling: %d, quant_table_id: %d\n", cinfo.horizontal_sampling,
               cinfo.vertical_sampling, cinfo.quant_table_id);
        // printf("");
    }

    spdlog::info("max horizontal sampling: {}, max vertical sampling: {}", images.sof.max_horizontal_sampling,
                 images.sof.max_vertical_sampling);
    return ret;
}

size_t JPEGDec::Parser_DHT(uint8_t *data_ptr) {
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
        printf("table type: %s, id: %d\n", ac_dc == 0 ? "DC" : "AC", id);
        for (int i = 0; i < 16; i++) {
            printf("%d ", height_info[i]);
        }
        printf("\n");
        len -= 17;
        std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> map;
        uint16_t code = 0;
        for (uint8_t h = 0; h < 16; h++) {
            for (int i = 0; i < height_info[h]; i++) {
                uint8_t symb = data_raw[0];
                data_raw += 1;
                map[h + 1][code] = symb;
                printf("len: %d\t", h + 1);
                printBits(2, &code);
                printf(" %#04X\n", symb);
                code += 1;
                len -= 1;
            }
            code <<= 1;
        }
        if (ac_dc == 0) {  // dc
            images.huffmanTable.dc_tables[id] = map;
        } else {  // ac
            images.huffmanTable.ac_tables[id] = map;
        }
    }
    return ret;
}

size_t JPEGDec::Parser_SOS(uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto len = big_endian_bytes2_uint(data_raw);
    auto ret = len;
    data_raw += 2;
    spdlog::info("SOS length: {}", len);

    auto &table_mapping_dc = images.table_mapping_dc;
    auto &table_mapping_ac = images.table_mapping_ac;
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

uint8_t JPEGDec::match_huffman(std::unordered_map<uint8_t, std::unordered_map<uint16_t, uint8_t>> &map) {

    uint8_t bit_len = 1;
    uint16_t code = 0;
    for (;;) {
        code = (code << 1);
        // code fetch a bit from tmp_byte
        code += bitStream->get_a_bit();

        auto iter = map[bit_len].find(code);
        if (iter != map[bit_len].end()) {
            // found in dc huffman table
            return iter->second;
        }
        bit_len += 1;
        if (bit_len > 16) {
            spdlog::error("cannot match huffman table");
            exit(1);
        }
    }
}
void printBlock(vector<float> &block) {
    printf("block value:\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%.3f ", block[i * 8 + j]);
        }
        printf("\n");
    }
}
size_t JPEGDec::read_block(uint8_t id, size_t block_idx) {
    logf << "channel: " << std::to_string(id) << " block: " << std::to_string(block_idx) << "" << std::endl;
    logf << std::to_string(bitStream->get_global_offset()) << " " << std::to_string(bitStream->get_bit_offset())
         << std::endl;
    auto &img = images;
    auto &dc_table = img.huffmanTable.dc_tables[img.table_mapping_dc[id]];
    auto &ac_table = img.huffmanTable.ac_tables[img.table_mapping_ac[id]];
    vector<float> block;
    block.resize(1);

    uint32_t pos_byte;
    uint32_t pos_bit;
    uint8_t code_len = match_huffman(dc_table);
    if (code_len == 0) {
        block[0] = img.last_dc[id];
    } else {
        img.last_dc[id] += bitStream->read_value(code_len);
        block[0] = img.last_dc[id];
    }
    // printf("dc value %f\n", block[0]);
    // exit(1);
    // read remain ac values
    uint8_t count = 1;
    while (count < 64) {
        code_len = match_huffman(ac_table);
        if (code_len == 0x00) {  // all zeros
            while (count < 64) {
                block.push_back(0.0);
                count++;
            }
        } else if (code_len == 0xF0) {  // sixteen zeros
            for (uint16_t i = 0; i < 16; i++) {
                block.push_back(0.0);
                count++;
            }
        } else {
            uint8_t zeros = code_len >> 4;
            float value = bitStream->read_value(code_len & 0x0F);
            for (uint16_t i = 0; i < zeros; i++) {
                block.push_back(0.0);
                count++;
            }
            block.push_back(value);
            count++;
        }
    }
    // printf("id: %d, block idx: %d\n", id, block_idx);
    // printBlock(block);

    img.mcus.mcu[id].push_back(block);

    return 0;
}

size_t JPEGDec::parser_mcu(uint16_t h_mcu_idx, uint16_t w_mcu_idx) {
    // printf("mcu h idx: %d, mcu w idx: %d\n", h_mcu_idx, w_mcu_idx);
    auto &img = images;
    for (uint8_t id = 0; id < 3; id++) {
        uint16_t height = img.sof.component_infos[id].vertical_sampling;
        uint16_t width = img.sof.component_infos[id].horizontal_sampling;

        for (uint16_t h = 0; h < height; h++) {
            for (uint16_t w = 0; w < width; w++) {
                read_block(id, h * width + w);
            }
        }
    }
    return 0;
}

size_t JPEGDec::Parser_MCUs(uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto sof_info = images.sof;
    auto image_width = sof_info.width;
    auto image_height = sof_info.height;

    uint16_t w = (image_width - 1) / (8 * sof_info.max_horizontal_sampling) + 1;
    uint16_t h = (image_height - 1) / (8 * sof_info.max_vertical_sampling) + 1;
    spdlog::info("width has {} MCUs, height has {} MCUs", w, h);
    images.mcus.w_mcu_num = w;
    images.mcus.h_mcu_num = h;

    spdlog::info("mcu start pos: {}", data_raw - data.data());
    bitStream = std::make_unique<BitStream>(data_raw, data.data());

    for (uint16_t i = 0; i < h; i++) {
        for (uint16_t j = 0; j < w; j++) {
            parser_mcu(i, j);
        }
    }
    return bitStream->get_ptr() - data_ptr;
}

void JPEGDec::Decoding_on_BlockOffset() {
    auto &img = images;
    if (img.blockpos.size() == 0) {
        spdlog::error("Please scan block boundaries first!");
        exit(1);
    }
    size_t mcu_has_blocks = 0;
    vector<size_t> mcu_has_blocks_prefix = {0, 0, 0};
    for (int id = 0; id < 3; id++) {
        mcu_has_blocks_prefix[id] = mcu_has_blocks;
        mcu_has_blocks +=
            img.sof.component_infos[id].horizontal_sampling * img.sof.component_infos[id].vertical_sampling;
    }
    uint16_t ww = img.mcus.w_mcu_num;
    uint16_t hh = img.mcus.h_mcu_num;
    for (uint16_t i = 0; i < hh; i++) {
        for (uint16_t j = 0; j < ww; j++) {
            for (int id = 0; id < 3; id++) {
                uint16_t height = img.sof.component_infos[id].vertical_sampling;              // h- block
                uint16_t width = img.sof.component_infos[id].horizontal_sampling;             // w - block
                auto &dc_table = images.huffmanTable.dc_tables[images.table_mapping_dc[id]];  // dc huffman table
                auto &ac_table = images.huffmanTable.ac_tables[images.table_mapping_ac[id]];  // ac huffman table

                for (uint16_t h = 0; h < height; h++) {
                    for (uint16_t w = 0; w < width; w++) {
                        const auto &blockpos =
                            images.blockpos[(i * ww + j) * mcu_has_blocks + mcu_has_blocks_prefix[id] + h * width + w];
                        uint32_t pos_in_byte = blockpos.first;
                        uint8_t pos_in_bit = blockpos.second;
                        BitStream bitStream(data.data() + pos_in_byte, pos_in_bit, data.data());

                        vector<float> block;
                        block.resize(1);
                        // for each block
                        // first 1 dc, then 63 ac
                        uint16_t tmp_codeword = bitStream.get_a_bit();
                        uint8_t tmp_codeword_len = 1;
                        for (;;) {  // match one dc huffman codeword
                            const auto &iter = dc_table[tmp_codeword_len].find(tmp_codeword);
                            if (iter != dc_table[tmp_codeword_len].end()) {  // found
                                uint8_t dc_value = iter->second;
                                if (dc_value == 0) {
                                    block[0] = img.last_dc[id];
                                } else {
                                    img.last_dc[id] += bitStream.read_value(dc_value);
                                    block[0] = img.last_dc[id];
                                }
                                break;
                            }
                            // not found
                            tmp_codeword = tmp_codeword << 1;
                            tmp_codeword += bitStream.get_a_bit();
                            tmp_codeword_len++;
                            if (tmp_codeword_len > 16) {
                                spdlog::error("huffman codeword should less than 16");
                                exit(1);
                            }
                        }
                        for (int ac_count = 0; ac_count < 63;) {  // match 63 ac values
                            uint16_t tmp_codeword = bitStream.get_a_bit();
                            uint8_t tmp_codeword_len = 1;
                            for (;;) {  // match one ac huffman codeword
                                const auto &iter = ac_table[tmp_codeword_len].find(tmp_codeword);
                                if (iter != ac_table[tmp_codeword_len].end()) {  // found
                                    uint8_t ac_value = iter->second;
                                    if (ac_value == 0x00) {  // all zeros
                                        for (; ac_count < 63; ac_count++) {
                                            block.push_back(0.0f);
                                        }
                                    } else if (ac_value == 0xF0) {  // sixteen zeros
                                        for (int i = 0; i < 16; i++) {
                                            block.push_back(0.0f);
                                        }
                                        ac_count += 16;
                                    } else {
                                        uint8_t zeros = ac_value >> 4;
                                        ac_count += zeros;
                                        for (uint8_t i = 0; i < zeros; i++) {
                                            block.push_back(0.0f);
                                        }
                                        float value = bitStream.read_value(ac_value & 0x0F);
                                        block.push_back(value);
                                        ac_count++;
                                    }
                                    break;
                                }
                                // not found
                                tmp_codeword = tmp_codeword << 1;
                                tmp_codeword += bitStream.get_a_bit();
                                tmp_codeword_len++;
                                if (tmp_codeword_len > 16) {
                                    spdlog::error("huffman codeword should less than 16");
                                    exit(1);
                                }
                            }
                        }
                        assert(block.size() == 64);
                        img.mcus.mcu[id].push_back(block);
                    }
                }
            }
        }
    }
}

size_t JPEGDec::Scan_MCUs(uint8_t *data_ptr) {
    uint8_t *data_raw = data_ptr;
    auto sof_info = images.sof;
    auto image_width = sof_info.width;
    auto image_height = sof_info.height;
    auto &img = images;

    uint16_t ww = (image_width - 1) / (8 * sof_info.max_horizontal_sampling) + 1;
    uint16_t hh = (image_height - 1) / (8 * sof_info.max_vertical_sampling) + 1;
    spdlog::info("width has {} MCUs, height has {} MCUs", ww, hh);
    images.mcus.w_mcu_num = ww;
    images.mcus.h_mcu_num = hh;

    spdlog::info("mcu start pos: {}", data_raw - data.data());

    BitStream bitStream(data_raw, data.data());

    size_t mcu_has_blocks = 0;
    vector<size_t> mcu_has_blocks_prefix = {0, 0, 0};
    for (int id = 0; id < 3; id++) {
        mcu_has_blocks_prefix[id] = mcu_has_blocks;
        mcu_has_blocks +=
            img.sof.component_infos[id].horizontal_sampling * img.sof.component_infos[id].vertical_sampling;
    }

    images.blockpos.resize(ww * hh * mcu_has_blocks);
    std::fill(images.blockpos.begin(), images.blockpos.end(), std::pair<size_t, uint8_t>{0, 0});

    for (uint16_t i = 0; i < hh; i++) {                                             // h - mcu
        for (uint16_t j = 0; j < ww; j++) {                                         // w - mcu
            for (uint8_t id = 0; id < 3; id++) {                                    // color channel
                uint16_t height = sof_info.component_infos[id].vertical_sampling;   // h- block
                uint16_t width = sof_info.component_infos[id].horizontal_sampling;  // w - block

                auto &dc_table = images.huffmanTable.dc_tables[images.table_mapping_dc[id]];  // dc huffman table
                auto &ac_table = images.huffmanTable.ac_tables[images.table_mapping_ac[id]];  // ac huffman table

                for (uint16_t h = 0; h < height; h++) {
                    for (uint16_t w = 0; w < width; w++) {
                        vector<float> block;
                        block.resize(1);

                        // uint32_t blockpos = bitStream.get_global_offset();
                        // blockpos << 3;
                        // blockpos += bitStream.get_bit_offset() & 0x07;

                        if (images.blockpos[(i * ww + j) * mcu_has_blocks + mcu_has_blocks_prefix[id] + h * width + w]
                                .first != 0) {
                            spdlog::error("error blockpos is not zero");
                        }
                        images.blockpos[(i * ww + j) * mcu_has_blocks + mcu_has_blocks_prefix[id] + h * width + w]
                            .first = bitStream.get_global_offset();
                        images.blockpos[(i * ww + j) * mcu_has_blocks + mcu_has_blocks_prefix[id] + h * width + w]
                            .second = bitStream.get_bit_offset();

                        bitStream.init(data.data() + bitStream.get_global_offset(), bitStream.get_bit_offset(),
                                       data.data());

                        // for each block
                        // first 1 dc, then 63 ac
                        uint16_t tmp_codeword = bitStream.get_a_bit();
                        uint8_t tmp_codeword_len = 1;
                        for (;;) {  // match one dc huffman codeword
                            const auto &iter = dc_table[tmp_codeword_len].find(tmp_codeword);
                            if (iter != dc_table[tmp_codeword_len].end()) {  // found
                                uint8_t dc_value = iter->second;
                                bitStream.skip_value(dc_value);
                                // if (dc_value == 0) {
                                //     block[0] = img.last_dc[id];
                                // } else {
                                //     img.last_dc[id] += bitStream.read_value(dc_value);
                                //     block[0] = img.last_dc[id];
                                // }
                                break;
                            }
                            // not found
                            tmp_codeword = tmp_codeword << 1;
                            tmp_codeword += bitStream.get_a_bit();
                            tmp_codeword_len++;
                            if (tmp_codeword_len > 16) {
                                spdlog::error("huffman codeword should less than 16");
                                exit(1);
                            }
                        }
                        for (int ac_count = 0; ac_count < 63;) {  // match 63 ac values
                            uint16_t tmp_codeword = bitStream.get_a_bit();
                            uint8_t tmp_codeword_len = 1;
                            for (;;) {  // match one ac huffman codeword
                                const auto &iter = ac_table[tmp_codeword_len].find(tmp_codeword);
                                if (iter != ac_table[tmp_codeword_len].end()) {  // found
                                    uint8_t ac_value = iter->second;
                                    // if (ac_value == 0x00) {  // all zeros
                                    //     for (; ac_count < 63; ac_count++) {
                                    //         block.push_back(0.0f);
                                    //     }
                                    // } else if (ac_value == 0xF0) {  // sixteen zeros
                                    //     for (int i = 0; i < 16; i++) {
                                    //         block.push_back(0.0f);
                                    //     }
                                    //     ac_count += 16;
                                    // } else {
                                    //     uint8_t zeros = ac_value >> 4;
                                    //     ac_count += zeros;
                                    //     for (uint8_t i = 0; i < zeros; i++) {
                                    //         block.push_back(0.0f);
                                    //     }
                                    //     float value = bitStream.read_value(ac_value & 0x0F);
                                    //     block.push_back(value);
                                    //     ac_count++;
                                    // }
                                    if (ac_value == 0x00) {  // all zeros
                                        ac_count = 63;
                                    } else if (ac_value == 0xF0) {  // sixteen zeros
                                        ac_count += 16;
                                    } else {
                                        uint8_t zeros = ac_value >> 4;
                                        ac_count += zeros;
                                        bitStream.skip_value(ac_value & 0x0F);
                                        ac_count++;
                                    }
                                    break;
                                }
                                // not found
                                tmp_codeword = tmp_codeword << 1;
                                tmp_codeword += bitStream.get_a_bit();
                                tmp_codeword_len++;
                                if (tmp_codeword_len > 16) {
                                    spdlog::error("huffman codeword should less than 16");
                                    exit(1);
                                }
                            }
                        }
                        // assert(block.size() == 64);
                        // img.mcus.mcu[id].push_back(block);
                    }
                }
            }
        }
    }
    return bitStream.get_ptr() - data_ptr;
}

void JPEGDec::Dequantize() {
    auto &sofinfo = images.sof;
    auto &comp_info = sofinfo.component_infos;
    auto &quant_tables = images.dqt_tables;
    for (uint8_t id = 0; id < 3; id++) {
        auto &c_info = comp_info[id];
        for (auto &vec : images.mcus.mcu[id]) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    vec[i * 8 + j] *= quant_tables[c_info.quant_table_id][i * 8 + j];
                }
            }
            // printBlock(vec);
            // exit(0);
        }
    }
}

const uint32_t zz_table[8][8] = {{0, 1, 5, 6, 14, 15, 27, 28},     {2, 4, 7, 13, 16, 26, 29, 42},
                                 {3, 8, 12, 17, 25, 30, 41, 43},   {9, 11, 18, 24, 31, 40, 44, 53},
                                 {10, 19, 23, 32, 39, 45, 52, 54}, {20, 22, 33, 38, 46, 51, 55, 60},
                                 {21, 34, 37, 47, 50, 56, 59, 61}, {35, 36, 48, 49, 57, 58, 62, 63}};

void JPEGDec::ZigZag() {
    auto &sofinfo = images.sof;
    auto &comp_info = sofinfo.component_infos;
    for (uint8_t id = 0; id < 3; id++) {
        auto &c_info = comp_info[id];
        for (auto &vec : images.mcus.mcu[id]) {
            vector<float> tmp_block = vec;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    vec[i * 8 + j] = tmp_block[zz_table[i][j]];
                }
            }
            // printBlock(vec);
            // exit(0);
        }
    }
}
static float cc2 = std::sqrt(0.5);
float cc(int i, int j) {
    if (i == 0 && j == 0) {
        return 0.5;
    } else if (i == 0 || j == 0) {
        return cc2;
    }
    return 1.0f;
}
void JPEGDec::IDCT() {
    auto &sofinfo = images.sof;
    auto &comp_info = sofinfo.component_infos;
    for (uint8_t id = 0; id < 3; id++) {
        auto &c_info = comp_info[id];
#pragma omp parallel for
        for (int idx_mcu = 0; idx_mcu < images.mcus.mcu[id].size(); idx_mcu++) {
            auto &vec = images.mcus.mcu[id][idx_mcu];
            vector<float> tmp_block(64, 0.0f);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    for (int x = 0; x < 8; x++) {
                        for (int y = 0; y < 8; y++) {
                            auto i_cos = std::cos((2 * i + 1) * M_PI / 16.0f * x);
                            auto j_cos = std::cos((2 * j + 1) * M_PI / 16.0f * y);
                            tmp_block[i * 8 + j] += cc(x, y) * vec[x * 8 + y] * i_cos * j_cos;
                        }
                    }
                    tmp_block[i * 8 + j] /= 4.0;
                }
            }
            vec = tmp_block;
            // printBlock(vec);
            // exit(0);
        }
    }
}

uint8_t chomp(float x) {
    if (x > 255.0) {
        return 255;
    } else if (x < 0.0) {
        return 0;
    }
    return std::round(x);
}

void JPEGDec::toRGB() {
    auto mcu_h = images.mcus.h_mcu_num;
    auto mcu_w = images.mcus.w_mcu_num;
    auto &sofinfo = images.sof;
    images.rgb.resize(images.sof.height * images.sof.width);

    for (int i = 0; i < sofinfo.height; i++) {
        for (int j = 0; j < sofinfo.width; j++) {  // pixel
            float YCbCr[3] = {0, 0, 0};
            int p2block_h = i / 8;
            int p2block_w = j / 8;
            for (int id = 0; id < 3; id++) {
                int p2mcu_h = p2block_h / sofinfo.component_infos[id].vertical_sampling;
                int p2mcu_w = p2block_w / sofinfo.component_infos[id].horizontal_sampling;
                float map_block_in_h =
                    1.0 * sofinfo.component_infos[id].vertical_sampling / sofinfo.max_vertical_sampling;
                float map_block_in_w =
                    1.0 * sofinfo.component_infos[id].horizontal_sampling / sofinfo.max_horizontal_sampling;
                int block_h = p2block_h * map_block_in_h;
                int block_w = p2block_w * map_block_in_w;
                // printf("pixel %d,%d maps to block %d,%d, ele %d,%d\n", i, j, block_h, block_w, (i % 8), (j % 8));
                YCbCr[id] = images.mcus.mcu[id][block_h * (mcu_w * sofinfo.component_infos[id].horizontal_sampling) +
                                                block_w][(i % 8) * 8 + (j % 8)];
            }
            RGBPix rgb;
            rgb.r = chomp(YCbCr[0] + 1.402 * YCbCr[2] + 128);
            rgb.g = chomp(YCbCr[0] - 0.34414 * YCbCr[1] - 0.71414 * YCbCr[2] + 128.0);
            rgb.b = chomp(YCbCr[0] + 1.772 * YCbCr[1] + 128.0);
            images.rgb[i * sofinfo.width + j] = rgb;
        }
    }
}
void JPEGDec::Dump(const string &fname) {
    std::ofstream of(fname, std::ios::binary);
    of << images.sof.height << std::endl;
    of << images.sof.width << std::endl;
    for (int i = 0; i < images.sof.height; i++) {
        for (int j = 0; j < images.sof.width; j++) {
            int off = i * images.sof.width + j;
            of << std::to_string(images.rgb[off].r) << " " << std::to_string(images.rgb[off].g) << " "
               << std::to_string(images.rgb[off].b) << "\n";
        }
    }
    of.close();
}
Image_struct JPEGDec::get_imgstruct() { return images; }
}  // namespace jpeg_dec
