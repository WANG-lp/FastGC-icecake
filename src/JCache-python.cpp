#include "../include/JCache.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace py = pybind11;
using namespace pybind11::literals;

namespace jcache {
string JPEGCacheClient::get_serialized_header(const std::string& filename) {
    string header_str;
    client->get(header_str, filename);
    assert(header_str.size() > 0);
    return header_str;
}
string JPEGCacheClient::get_serialized_header_ROI(const std::string& filename, int32_t offset_x, int32_t offset_y,
                                                  int32_t roi_w, int32_t roi_h) {
    string header_str;
    client->getWithROI(header_str, filename, offset_x, offset_y, roi_w, roi_h);
    assert(header_str.size() > 0);
    return header_str;
}
string JPEGCacheClient::get_serialized_raw_file(const std::string& filename) {
    string ret;
    client->getRAW(ret, filename);
    assert(ret.size() > 0);
    return ret;
}
}  // namespace jcache

PYBIND11_MODULE(pjcache, m) {
    m.doc() = "pybind11 JCacheClient plugin";
    py::class_<jcache::JPEGCacheClient>(m, "JPEGCacheClient")
        .def(py::init<string, int>())
        .def("put", (int (jcache::JPEGCacheClient::*)(const std::string&)) & jcache::JPEGCacheClient::put,
             "put a file to remote cache server", "filename"_a)
        .def(
            "get_serialized_header",
            [](jcache::JPEGCacheClient& jc, const string& filename) {
                return py::bytes(jc.get_serialized_header(filename));
            },
            "get a parsed file from remote cache server", "filename"_a)
        .def(
            "get_serialized_header_ROI",
            [](jcache::JPEGCacheClient& jc, const string& filename, int32_t offset_x, int32_t offset_y, int32_t width,
               int32_t height) {
                return py::bytes(jc.get_serialized_header_ROI(filename, offset_x, offset_y, width, height));
            },
            "get a parsed file from remote cache server with ROI", "filename"_a, "offset_x"_a, "offset_y"_a, "width"_a,
            "height"_a)
        .def(
            "get_raw_file",
            [](jcache::JPEGCacheClient& jc, const string& filename) {
                return py::bytes(jc.get_serialized_raw_file(filename));
            },
            "get a file from remote cache server", "filename"_a);
}
