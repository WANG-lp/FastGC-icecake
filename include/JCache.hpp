#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "jpeg_decoder_def.hpp"

using std::string;
using std::vector;

namespace jcache {
class JCache {
   public:
    JCache();
    ~JCache() = default;

    bool putJPEG(const vector<uint8_t> &image, const string &filename);
    bool putJPEG(const string &filename);
    JPEG_HEADER *getHeader(const string &filename);
    JPEG_HEADER *getHeaderwithCrop(const string &filename, int offset_x, int offset_y, int roi_width, int roi_height);
   private:
    std::unordered_map<string, JPEG_HEADER> map_;
};
}  // namespace jcache