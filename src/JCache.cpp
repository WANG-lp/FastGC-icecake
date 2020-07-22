#include "../include/JCache.hpp"
#include "../include/jpeg_decoder.hpp"
#include "../include/jpeg_decoder_export.h"

#include <spdlog/spdlog.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <fstream>
#include <random>

template <class Archive>
void serialize(Archive &archive, JPEG_HEADER &m) {
    archive(m.dqt_table, m.sof0, m.dht, m.sos_first_part, m.sos_second_part, m.blockpos_compact, m.blocks_num, m.width,
            m.height, m.status);
}

namespace jcache {

JPEG_HEADER *deserialization_header(const string &str) {
    std::stringstream iss(str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);
    JPEG_HEADER *header = static_cast<JPEG_HEADER *>(create_jpeg_header());
    iarchive(*header);
    // printf("deserialization ok. height: %d, width: %d\n", header->height, header->width);
    return header;
}

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

inline int nround(int n, int multiple = 8) {
    n = ((n + multiple / 2) / multiple) * multiple;
    return n;
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

    set_parameters(0, 0.08, 1.0, 3.0 / 4, 4.0 / 3);
    thread_count = 0;
};
JCache::~JCache() {
    if (server_handle) {
        server_handle->stop();
    }
    if (isStarted) {
        server_tid.join();
    }
}

void JCache::set_parameters(int seed, float s1, float s2, float r1, float r2) {
    this->seed = seed;
    scale1 = s1;
    scale2 = s2;
    ratio1 = r1;
    ratio2 = r2;
    aspect_ratio1 = std::log(r1);
    aspect_ratio2 = std::log(r2);
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
    map_[filename] = {header, string(image_raw, image_raw + len)};
    return true;
}
bool JCache::putJPEG(const vector<uint8_t> &image, const string &filename) {
    jpeg_dec::JPEGDec dec(image);
    dec.Parser();
    auto header = dec.get_header();
    if (header.status != 1) {
        return false;
    }
    map_[filename] = {header, string(image.begin(), image.end())};
    return true;
}
bool JCache::putJPEG(const string &filename) {
    jpeg_dec::JPEGDec dec(filename);
    printf("put\n");
    dec.Parser();
    auto header = dec.get_header();
    if (header.status != 1) {
        return false;
    }
    printf("%d\n", header.blocks_num);
    auto img_data = dec.get_image_data();
    map_[filename] = {header, string(img_data.begin(), img_data.end())};
    return true;
}

JPEG_HEADER *JCache::getHeader(const string &filename) {
    auto e = map_.find(filename);
    if (e != map_.end()) {
        JPEG_HEADER *ret = static_cast<JPEG_HEADER *>(create_jpeg_header());
        auto &header = e->second.first;
        ret->blockpos_compact = header.blockpos_compact;
        ret->blocks_num = header.blocks_num;
        ret->dht = header.dht;
        ret->dqt_table = header.dqt_table;
        ret->height = header.height;
        ret->sof0 = header.sof0;
        ret->sos_first_part = header.sos_first_part;
        ret->sos_second_part = header.sos_second_part;
        ret->status = header.status;
        ret->width = header.width;
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
    auto &header = e->second.first;
    return static_cast<JPEG_HEADER *>(onlineROI(&header, offset_x, offset_y, roi_width, roi_height));
}

JPEG_HEADER *JCache::getHeaderRandomCrop(const string &filename) {
    static thread_local int tid = thread_count++;
    static thread_local std::mt19937 generator(seed + tid);

    auto e = map_.find(filename);
    if (e != map_.end()) {
        auto &header = e->second.first;
        int width = header.width;
        int height = header.height;
        size_t area = width * height;

        bool found = false;
        std::uniform_real_distribution<float> distribution(scale1, scale2);
        std::uniform_real_distribution<float> distribution2(aspect_ratio1, aspect_ratio2);

        size_t w, h, i, j;
        for (int iter = 0; iter < 10; iter++) {  // try 10 times
            size_t target_area = area * distribution(generator);
            float a_ratio = std::exp(distribution2(generator));

            w = size_t(std::sqrt(target_area * a_ratio));
            h = size_t(std::sqrt(target_area / a_ratio));

            w = nround(w);
            h = nround(h);

            if (w > 0 && w <= width && h > 0 && h <= height) {
                found = true;
                std::uniform_int_distribution<int> distribution3(0, width - w);
                std::uniform_int_distribution<int> distribution4(0, height - h);
                i = distribution3(generator);
                j = distribution4(generator);
                i = nround(i);
                j = nround(j);
                break;
            }
        }
        if (!found) {  // we crop from upper-left and ignore ratio
            printf("cannot get random i,j,w,h\n");
            i = j = 0;
            h = nround(height / 2);
            w = nround(width / 2);
        }
        return static_cast<JPEG_HEADER *>(onlineROI(&header, i, j, w, h));
    }
    return nullptr;
}

string JCache::getRAWData(const string &filename) {
    auto e = map_.find(filename);
    if (e != map_.end()) {
        return e->second.second;
    }
    return "";
}

JPEGCacheHandler::JPEGCacheHandler(){

};

void JPEGCacheHandler::setJCache(JCache *jc) { this->jcache = jc; }
int32_t JPEGCacheHandler::set_parameters(const int32_t seed, const double s1, const double s2, const double r1,
                                         const double r2) {
    jcache->set_parameters(seed, s1, s2, r1, r2);
    return 0;
}
void JPEGCacheHandler::get(std::string &_return, const std::string &filename) {
    _return.resize(0);
    auto header = std::unique_ptr<JPEG_HEADER>(jcache->getHeader(filename));
    if (header) {
        std::stringstream ss(std::ios::out | std::ios::binary);
        cereal::BinaryOutputArchive archive(ss);
        archive(*header);
        _return = ss.str();
        // spdlog::info("serialized_str len: {}", _return.size());
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
        // spdlog::info("serialized_str len: {}", _return.size());
    }
}

void JPEGCacheHandler::getWithRandomCrop(std::string &_return, const std::string &filename) {
    _return.resize(0);
    auto header = std::unique_ptr<JPEG_HEADER>(jcache->getHeaderRandomCrop(filename));
    if (header) {
        std::stringstream ss(std::ios::out | std::ios::binary);
        cereal::BinaryOutputArchive archive(ss);
        archive(*header);
        _return = ss.str();
        // spdlog::info("serialized_str len: {}", _return.size());
    }
}

void JPEGCacheHandler::getRAW(std::string &_return, const std::string &filename) {
    _return = jcache->getRAWData(filename);
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

int32_t JPEGCacheClient::set_parameters(const int32_t seed, const double s1, const double s2, const double r1,
                                        const double r2) {
    return client->set_parameters(seed, s1, s2, r1, r2);
}

JPEG_HEADER *JPEGCacheClient::get(const std::string &filename) {
    string header_str;
    client->get(header_str, filename);
    assert(header_str.size() > 0);
    printf("serialize size: %f\n", header_str.size() / 1024.0);

    std::stringstream iss(header_str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);

    JPEG_HEADER *header = static_cast<JPEG_HEADER *>(create_jpeg_header());
    iarchive(*header);
    return header;
}
JPEG_HEADER *JPEGCacheClient::getWithROI(const std::string &filename, int32_t offset_x, int32_t offset_y, int32_t roi_w,
                                         int32_t roi_h) {
    string header_str;
    client->getWithROI(header_str, filename, offset_x, offset_y, roi_w, roi_h);
    assert(header_str.size() > 0);
    printf("serialize size: %f\n", header_str.size() / 1024.0);

    std::stringstream iss(header_str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);
    JPEG_HEADER *header = static_cast<JPEG_HEADER *>(create_jpeg_header());
    iarchive(*header);
    return header;
}

JPEG_HEADER *JPEGCacheClient::getWithRandomCrop(const std::string &filename) {
    string header_str;
    client->getWithRandomCrop(header_str, filename);
    assert(header_str.size() > 0);
    printf("serialize size: %f\n", header_str.size() / 1024.0);

    std::stringstream iss(header_str, std::ios::in | std::ios::binary);
    cereal::BinaryInputArchive iarchive(iss);
    JPEG_HEADER *header = static_cast<JPEG_HEADER *>(create_jpeg_header());
    iarchive(*header);
    return header;
}

string JPEGCacheClient::get_serialized_header(const std::string &filename) {
    string header_str;
    client->get(header_str, filename);
    assert(header_str.size() > 0);
    return header_str;
}
string JPEGCacheClient::get_serialized_header_ROI(const std::string &filename, int32_t offset_x, int32_t offset_y,
                                                  int32_t roi_w, int32_t roi_h) {
    string header_str;
    client->getWithROI(header_str, filename, offset_x, offset_y, roi_w, roi_h);
    assert(header_str.size() > 0);
    return header_str;
}
string JPEGCacheClient::get_serialized_header_random_crop(const std::string &filename) {
    string header_str;
    client->getWithRandomCrop(header_str, filename);
    assert(header_str.size() > 0);
    return header_str;
}
string JPEGCacheClient::get_serialized_raw_file(const std::string &filename) {
    string ret;
    client->getRAW(ret, filename);
    assert(ret.size() > 0);
    return ret;
}

int32_t JPEGCacheClient::put(const std::string &filename, const std::string &content) {
    return client->put(filename, content);
}
int32_t JPEGCacheClient::put(const std::string &filename, const uint8_t *content_raw, size_t content_len) {
    std::string tmp(content_raw, content_raw + content_len);
    return client->put(filename, tmp);
}
int32_t JPEGCacheClient::put(const std::string &filename, const vector<uint8_t> &content) {
    return put(filename, content.data(), content.size());
}
int32_t JPEGCacheClient::put(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::ate | std::ios::binary);
    if (!ifs.good()) {
        printf("error while opening file %s\n", filename.c_str());
        return 1;
    }
    size_t fsize = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    vector<uint8_t> tmp_vec(fsize);
    ifs.read(reinterpret_cast<char *>(tmp_vec.data()), fsize);

    return put(filename, tmp_vec);
}

}  // namespace jcache