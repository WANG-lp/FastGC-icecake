#include "../include/JCache.hpp"
#include "../include/jpeg_decoder.hpp"
#include "../include/jpeg_decoder_export.h"

#include <spdlog/spdlog.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

template <class Archive>
void serialize(Archive &archive, JPEG_HEADER &m) {
    archive(m.dqt_table, m.sof0, m.dht, m.sos_first_part, m.sos_second_part, m.blockpos_compact, m.blocks_num, m.width,
            m.height, m.status);
}

namespace jcache {

inline uint16_t big_endian_bytes2_uint(const void *data) {
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

inline void bytes2_big_endian_uint(uint16_t len, uint8_t *target_ptr) {
    unsigned char *b = (unsigned char *) target_ptr;
    unsigned char *p = (unsigned char *) &len;
#ifdef _BIG_ENDIAN
    b[0] = p[0];
    b[1] = p[1];
#else
    b[0] = p[1];
    b[1] = p[0];
#endif
}

JCache::JCache(int port) {
    jpegcachehandler = std::make_shared<JPEGCacheHandler>();
    jpegcachehandler->setJCache(this);
    std::shared_ptr<at::server::TProcessor> processor(new JPEGCacheProcessor(jpegcachehandler));
    std::shared_ptr<atp::TProtocolFactory> protocolFactory(new atp::TBinaryProtocolFactory());
    auto threadManager = at::concurrency::ThreadManager::newSimpleThreadManager(16);
    auto threadFactory = std::make_shared<at::concurrency::ThreadFactory>();
    threadManager->threadFactory(threadFactory);
    threadManager->start();
    std::shared_ptr<att::TTransportFactory> transportFactory(new att::TFramedTransportFactory());

    server_transport = std::make_shared<att::TServerSocket>("0.0.0.0", port);
    server_handle = std::make_shared<at::server::TThreadPoolServer>(processor, server_transport, transportFactory,
                                                                    protocolFactory, threadManager);
};
JCache::~JCache() {
    server_handle->stop();
    if (isStarted) {
        server_tid.join();
    }
}

void JCache::startServer() { server_tid = std::thread(std::bind(&JCache::server_func, this)); }
void JCache::serve() { server_func(); }
void JCache::server_func() {
    isStarted = true;
    server_handle->serve();
}

bool JCache::putJPEG(const uint8_t *image_raw, size_t len, const string &filename) {
    jpeg_dec::JPEGDec dec(image_raw, len);
    dec.Parser();
    auto header = dec.get_header();
    if (header.status != 1) {
        return false;
    }
    map_[filename] = header;
    return true;
}
bool JCache::putJPEG(const vector<uint8_t> &image, const string &filename) {
    jpeg_dec::JPEGDec dec(image);
    dec.Parser();
    auto header = dec.get_header();
    if (header.status != 1) {
        return false;
    }
    map_[filename] = header;
    return true;
}
bool JCache::putJPEG(const string &filename) {
    jpeg_dec::JPEGDec dec(filename);
    dec.Parser();
    auto header = dec.get_header();
    if (header.status != 1) {
        return false;
    }
    map_[filename] = header;
    return true;
}
JPEG_HEADER *JCache::getHeader(const string &filename) {
    auto e = map_.find(filename);
    if (e != map_.end()) {
        JPEG_HEADER *ret = static_cast<JPEG_HEADER *>(create_jpeg_header());
        ret->blockpos_compact = e->second.blockpos_compact;
        ret->blocks_num = e->second.blocks_num;
        ret->dht = e->second.dht;
        ret->dqt_table = e->second.dqt_table;
        ret->height = e->second.height;
        ret->sof0 = e->second.sof0;
        ret->sos_first_part = e->second.sos_first_part;
        ret->sos_second_part = e->second.sos_second_part;
        ret->status = e->second.status;
        ret->width = e->second.width;
        return ret;
    }
    return nullptr;
}
JPEG_HEADER *JCache::getHeaderwithCrop(const string &filename, int offset_x, int offset_y, int roi_width,
                                       int roi_height) {
    auto e = map_.find(filename);
    if (e == map_.end()) {
        return nullptr;
    }
    auto header = &e->second;
    return static_cast<JPEG_HEADER *>(onlineROI(header, offset_x, offset_y, roi_width, roi_height));
}

JPEGCacheHandler::JPEGCacheHandler(){

};

void JPEGCacheHandler::setJCache(JCache *jc) { this->jcache = jc; }

void JPEGCacheHandler::get(std::string &_return, const std::string &filename) {
    _return.resize(0);
    auto header = std::unique_ptr<JPEG_HEADER>(jcache->getHeader(filename));
    if (header) {
        std::stringstream ss(std::ios::out | std::ios::binary);
        cereal::BinaryOutputArchive archive(ss);
        archive(*header);
        _return = ss.str();
        spdlog::info("serialized_str len: {}", _return.size());
    }
}

void JPEGCacheHandler::getWithROI(std::string &_return, const std::string &filename, const int32_t offset_x,
                                  const int32_t offset_y, const int32_t roi_w, const int32_t roi_h) {
    _return.resize(0);
    auto header = std::unique_ptr<JPEG_HEADER>(jcache->getHeaderwithCrop(filename, offset_x, offset_y, roi_w, roi_h));
    if (header) {
        std::stringstream ss(std::ios::out | std::ios::binary);
        cereal::BinaryOutputArchive archive(ss);
        archive(*header);
        _return = ss.str();
        spdlog::info("serialized_str len: {}", _return.size());
    }
}

int32_t JPEGCacheHandler::put(const std::string &filename, const std::string &content) {
    bool ret = jcache->putJPEG(reinterpret_cast<const uint8_t *>(content.data()), content.size(), filename);
    return ret ? 0 : 1;
}

JPEGCacheClient::JPEGCacheClient(const string &host, int port) {
    socket = std::make_shared<att::TSocket>(host, port);
    transport = std::shared_ptr<att::TFramedTransport>(new att::TFramedTransport(socket));
    protocol = std::shared_ptr<atp::TProtocol>(new atp::TBinaryProtocol(transport));
    client = std::make_shared<JPEGCache::JPEGCacheClient>(protocol);
    transport->open();
};
JPEGCacheClient::~JPEGCacheClient() { transport->close(); };

JPEG_HEADER JPEGCacheClient::get(const std::string &filename) {
    string header_str;
    client->get(header_str, filename);
    assert(header_str.size() > 0);
    printf("serialize size: %f\n", header_str.size() / 1024.0);

    std::stringstream iss(header_str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);

    JPEG_HEADER header;
    iarchive(header);
    return header;
}
JPEG_HEADER JPEGCacheClient::getWithROI(const std::string &filename, int32_t offset_x, int32_t offset_y, int32_t roi_w,
                                        int32_t roi_h) {
    JPEG_HEADER header;
    string header_str;
    client->getWithROI(header_str, filename, offset_x, offset_y, roi_w, roi_h);
    assert(header_str.size() > 0);
    printf("serialize size: %f\n", header_str.size() / 1024.0);

    std::stringstream iss(header_str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);
    iarchive(header);
    return header;
}
int32_t JPEGCacheClient::put(const std::string &filename, const std::string &content) {
    return client->put(filename, content);
}
int32_t JPEGCacheClient::put(const std::string &filename, const uint8_t *content_raw, size_t content_len) {
    std::string tmp(content_raw, content_raw + content_len);
    return client->put(filename, tmp);
}

}  // namespace jcache