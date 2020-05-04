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
    for (const auto &pos : imgs.blockpos) {
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
    for (int i = 0; i < 10; i++) {
        uint32_t pos = jpeg_dec.get_imgstruct().blockpos[i];
        printf("block %d, start offset: %d, bit: %d\n", i, pos >> 3, pos & 0x07);
    }
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
    jpeg_dec.Dump("/tmp/out.bin");
    spdlog::info("parse time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(t1 - s_t).count());
    spdlog::info("huffman decoding time: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(t3 - s_t).count());
    spdlog::info("IDCT time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(other2 - other1).count());
    spdlog::info("Other time: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1 - (other2 - other1)).count());
}

int main(int argc, char **argv) {
    // get_block_pos(argv[1], argv[2]);
    // test_jpegdec_scan_block(argv[1]);
    test_jpegdec(argv[1]);
    return 0;
}
