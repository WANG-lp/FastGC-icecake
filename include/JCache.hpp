#pragma once
#include <memory>
#include <string>
#include <thread>
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
class JPEGCacheHandler;

class JCache {
   public:
    JCache(int port = 8090);
    ~JCache();
    bool putJPEG(const uint8_t *image_raw, size_t len, const string &filename);
    bool putJPEG(const vector<uint8_t> &image, const string &filename);
    bool putJPEG(const string &filename);
    JPEG_HEADER *getHeader(const string &filename);
    JPEG_HEADER *getHeaderwithCrop(const string &filename, int offset_x, int offset_y, int roi_width, int roi_height);

    void startServer();
    void serve();

   private:
    void server_func();

    std::unordered_map<string, JPEG_HEADER> map_;
    std::thread server_tid;
    bool isStarted = false;
    std::shared_ptr<att::TServerSocket> server_transport;
    std::shared_ptr<at::server::TThreadPoolServer> server_handle;
    std::shared_ptr<JPEGCacheHandler> jpegcachehandler;
};

class JPEGCacheHandler : virtual public JPEGCacheIf {
   public:
    JPEGCacheHandler();
    ~JPEGCacheHandler() = default;
    void get(std::string &_return, const std::string &filename);
    void getWithROI(std::string &_return, const std::string &filename, const int32_t offset_x, const int32_t offset_y,
                    const int32_t roi_w, const int32_t roi_h);
    int32_t put(const std::string &filename, const std::string &content);

    void setJCache(JCache *jcache);

   private:
    JCache *jcache;
};

class JPEGCacheClient {
   public:
    JPEGCacheClient(const string &host, int port);
    ~JPEGCacheClient();

    JPEG_HEADER get(const std::string &filename);
    JPEG_HEADER getWithROI(const std::string &filename, int32_t offset_x, int32_t offset_y, int32_t roi_w,
                           int32_t roi_h);
    int32_t put(const std::string &filename, const std::string &content);
    int32_t put(const std::string &filename, const uint8_t *content_raw, size_t content_len);

   private:
    std::shared_ptr<att::TSocket> socket;
    std::shared_ptr<att::TTransport> transport;
    std::shared_ptr<atp::TProtocol> protocol;
    std::shared_ptr<JPEGCache::JPEGCacheClient> client;
};

}  // namespace jcache