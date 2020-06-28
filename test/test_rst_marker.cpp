#include <atomic>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "../3rd-deps/spdlog/include/spdlog/spdlog.h"

using std::string;
using std::vector;

vector<string> filelist;
const string BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/";

std::atomic<size_t> has_rst_count(0);

vector<size_t> rst_marker_distribution;
size_t max_rst_interval = 0;

static uint16_t big_endian_bytes2_uint(void *data) {
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

bool has_rst_marker(const string &fname) {
    std::ifstream inf(fname, std::ios::binary | std::ios::ate);
    if (!inf.good()) {
        spdlog::error("cannot open file {}", fname);
        return false;
    }

    std::streamsize size = inf.tellg();
    inf.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    inf.read(buffer.data(), size);
    // spdlog::info("size: {}", buffer.size());

    unsigned char c1 = buffer[0];
    for (size_t i = 1; i < size; i++) {
        unsigned char c2 = buffer[i];
        if (c1 == 0xFF && c2 == 0xDD) {
            uint16_t tmp = big_endian_bytes2_uint(buffer.data() + i + 1);
            assert(tmp == 4);
            tmp = big_endian_bytes2_uint(buffer.data() + i + 3);
            if (rst_marker_distribution.size() <= tmp) {
                rst_marker_distribution.resize(tmp + 1, 0);
            }
            rst_marker_distribution[tmp]++;
            return true;
        }
        c1 = c2;
    }
    return false;
}

void test_rst_marker() {
    for (size_t i = 0; i < filelist.size(); i++) {

        if (has_rst_marker(BASE_DIR + filelist[i])) {
            has_rst_count++;
            // spdlog::info("{}", BASE_DIR + filelist[i]);
        }
    }
}

int main(int argc, char **argv) {
    {
        std::ifstream inf(argv[1]);
        assert(inf.good());
        std::string line;
        while (std::getline(inf, line)) {
            filelist.push_back(line);
        }
    }

    spdlog::info("filelist has {} files", filelist.size());

    test_rst_marker();

    spdlog::info("total file: {}, has_rst_marker: {}", filelist.size(), has_rst_count);
    for (int i = 0; i < rst_marker_distribution.size(); i++) {
        printf("rst_interval: %d: %d\n", i, rst_marker_distribution[i]);
    }
    return 0;
}