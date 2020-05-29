#include "../include/jpeg_decoder.hpp"

#include <spdlog/spdlog.h>
#include <fstream>
#include <string>
#include <vector>
using std::string;
using std::vector;
const string BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/";
void get_block_pos(const string &fname, const string &fout) {
    jpeg_dec::JPEGDec jpeg_dec(fname);
    auto s_t = jpeg_dec::get_wall_time();
    jpeg_dec.Parser();
    auto t1 = jpeg_dec::get_wall_time();
    // jpeg_dec.Dequantize();
    // jpeg_dec.ZigZag();
    // jpeg_dec.IDCT();
    // jpeg_dec.toRGB();
    auto t2 = jpeg_dec::get_wall_time();
    auto imgs = jpeg_dec.get_imgstruct();
    int count = 0;

    std::ofstream of(fout, std::ios::app | std::ofstream::out);
    of << "==========" << std::endl;
    of << fname << std::endl;
    for (const auto &pos : imgs.recordFileds.blockpos) {
        char str[1024];
        memset(str, 0, sizeof(str));
        // int n = sprintf(str, "block: %d, dc: %d %d, ac: %d %d\n", count++, pos.dc_pos_byte, pos.dc_pos_bit,
        //                 pos.ac_pos_byte, pos.ac_pos_bit);
        // of.write(str, n);
    }
    of << std::endl;

    // jpeg_dec.Dump(0, "/tmp/out.bin");

    spdlog::info("parse time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(t1 - s_t).count());
    spdlog::info("calc time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(t2 - s_t).count());
}
void test_jpegdec_scan_block(const string &fname) {
    jpeg_dec::JPEGDec jpeg_dec(fname);
    auto s_t = jpeg_dec::get_wall_time();
    jpeg_dec.Parser();
    auto t1 = jpeg_dec::get_wall_time();
    spdlog::info("parse time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(t1 - s_t).count());

    if (fname != "/tmp/test_com.jpeg")
        jpeg_dec.WriteBoundarytoFile("/tmp/test_com.jpeg");

    spdlog::info("block num: {}", jpeg_dec.get_imgstruct().recordFileds.blockpos.size());
    for (int i = 0; i < jpeg_dec.get_imgstruct().recordFileds.blockpos.size(); i++) {
        auto pos = jpeg_dec.get_imgstruct().recordFileds.blockpos[i];
        printf("block %d, start offset: %ld, bit: %d\n", i, pos.first, pos.second);
    }
    if (fname != "/tmp/test_com.jpeg")
        jpeg_dec.WriteBoundarytoFile("/tmp/test_com.jpeg");
}
void test_jpegdec(const string &fname) {
    jpeg_dec::JPEGDec jpeg_dec(fname);
    auto s_t = jpeg_dec::get_wall_time();
    jpeg_dec.Parser();
    auto t1 = jpeg_dec::get_wall_time();
    jpeg_dec.Decoding_on_BlockOffset();
    auto t3 = jpeg_dec::get_wall_time();
    jpeg_dec.Dequantize();
    jpeg_dec.ZigZag();
    auto other1 = jpeg_dec::get_wall_time();
    jpeg_dec.IDCT();
    auto other2 = jpeg_dec::get_wall_time();
    jpeg_dec.toRGB();
    auto t2 = jpeg_dec::get_wall_time();
    jpeg_dec.Dump("/tmp/out.bmp");
    spdlog::info("parse time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(t1 - s_t).count());
    spdlog::info("huffman decoding time: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(t3 - s_t).count());
    spdlog::info("IDCT time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(other2 - other1).count());
    spdlog::info("Other time: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1 - (other2 - other1)).count());
    if (fname != "/tmp/test_com.jpeg")
        jpeg_dec.WriteBoundarytoFile("/tmp/test_com.jpeg");
}
void test_bitstream() {
    vector<std::pair<size_t, uint8_t>> numbers = {{111, 1}, {321, 2}, {563, 3}, {884, 4}};
    jpeg_dec::OBitStream obs;
    for (const auto &e : numbers) {
        size_t off = e.first;
        printf("pair_first %d\n", off);
        size_t mask = 1 << 28;
        for (int i = 0; i < 29; i++) {
            obs.write_bit(off & mask ? 1 : 0);
            mask = mask >> 1;
        }
        uint8_t off_bit = e.second;
        mask = 1 << 2;
        for (int i = 0; i < 3; i++) {
            obs.write_bit(off_bit & mask ? 1 : 0);
            mask = mask >> 1;
        }
    }
    auto data_vec = obs.get_data();
    printf("data len %d\n", data_vec.size());
    for (int i = 0; i < data_vec.size(); i++) {
        printf("%d ", data_vec[i]);
    }
    printf("\n");
    printf("%p\n", data_vec.data());
    jpeg_dec::BitStream ibs(data_vec.data(), false);
    for (size_t c = 0; c < numbers.size(); c++) {
        size_t off = 0;
        for (int i = 0; i < 29; i++) {
            off <<= 1;
            off += ibs.get_a_bit();
        }
        uint8_t off_bit = 0;
        for (int i = 0; i < 3; i++) {
            off_bit <<= 1;
            off_bit += ibs.get_a_bit();
        }
        spdlog::info("off: {}, bit: {}", off, off_bit);
    }
}
int main(int argc, char **argv) {
    // get_block_pos(argv[1], argv[2]);
    // test_jpegdec_scan_block(argv[1]);
    test_jpegdec(argv[1]);
    // test_bitstream();
    return 0;
}
