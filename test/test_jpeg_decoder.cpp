#include "../include/jpeg_decoder.hpp"

#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
const string BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/";
void test_jpegdec(const string &fname) {
    jpeg_dec::JPEGDec jpeg_dec(fname);
    jpeg_dec.Parser(0);
    jpeg_dec.Dequantize(0);
    jpeg_dec.ZigZag(0);
    jpeg_dec.IDCT(0);
    jpeg_dec.toRGB(0);
    jpeg_dec.Dump(0, "/tmp/out.bin");
}

void get_jpeg_config(const vector<string> &fnames, const string &fout) {
    std::ofstream fo(fout);
    for (const auto &f : fnames) {
        jpeg_dec::JPEGDec jpeg_dec(f);
        jpeg_dec.Parser(0);
        const auto &imginfo = jpeg_dec.get_imgInfo(0);
        fo << f << ": \n";
        for (int i = 0; i < 3; i++) {
            fo << std::to_string(imginfo.sof.component_infos[i].horizontal_sampling) << " "
               << std::to_string(imginfo.sof.component_infos[i].vertical_sampling) << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    vector<string> fnames;
    std::ifstream fin(argv[1]);
    std::string str;
    int count = 0;
    while (std::getline(fin, str)) {
        fnames.push_back(BASE_DIR + str);
        if (count++ > 10)
            break;
    }
    get_jpeg_config(fnames, "finfo.txt");
    return 0;
}
