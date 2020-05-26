#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "jpeg_decoder_def.hpp"

#include <thrift/concurrency/ThreadFactory.h>
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/transport/TSocket.h>

#include "../src/gen-cpp/JPEGCache.h"

using std::string;
using std::vector;

namespace at = ::apache::thrift;
namespace atp = ::apache::thrift::protocol;
namespace att = ::apache::thrift::transport;
using namespace ::JPEGCache;

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

class JPEGCacheHandler : virtual public JPEGCacheIf {
   public:
    JPEGCacheHandler(){};
    ~JPEGCacheHandler() = default;
    void get(std::string &_return, const std::string &filename){

    };
    void getWithROI(std::string &_return, const std::string &filename, const int32_t offset_x, const int32_t offset_y,
                    const int32_t roi_w, const int32_t roi_h) {}
    int32_t put(const std::string &filenames) { return 0; }
};

}  // namespace jcache