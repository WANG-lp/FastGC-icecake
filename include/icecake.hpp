
#pragma once
#include <parallel_hashmap/phmap.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <string>
#include <vector>
#include "dlpack.h"

using std::string;
using std::vector;

int sub(int a, int b);
std::string csayhello(const std::string& str);
int sub_device(int a, int b);

namespace pybind11 {
class capsule;
class array;
}  // namespace pybind11

namespace icecake {
class tensor {
   public:
    tensor();
    ~tensor();
    bool to_cuda(int device_id = 0);
    bool to_cuda_from_raw(int device_id = 0, void* raw = nullptr, size_t raw_len = 0);
    std::pair<size_t, size_t> get_pos(const string& fid);
    // ---------data area--------------
    int tensor_type_;  // 0->host, 1->cuda
    int device_id_;
    void* data_;  // data_pointer;
    size_t data_size_;
};

typedef struct {
    size_t total_write;
    size_t origin_size;
    size_t total_read;
} statistics;

class GPUCache {
   public:
    int total_cuda_device = 0;

    GPUCache(size_t alloc_size);
    ~GPUCache();

    void write_to_device_memory(const string& fid, const char* meta, size_t meta_length, const char* blockRAW,
                                size_t length, int device);
    char* read_from_device_memory(const string& fid, size_t* length);
    char* read_from_device_memory(const string& fid);
    bool put_dltensor_to_device_memory(const string& fid, DLManagedTensor* dltensor);
    DLManagedTensor* get_dltensor_from_device(const string& fid, int device);
    std::pair<size_t, size_t> get_pos(const string& fid);

    bool put_numpy_array(const string& fid, pybind11::array narray);
    bool put_dltensor(const string& fid, pybind11::capsule capsule);
    pybind11::capsule get_dltensor(const string& fid, int device);

   private:
    statistics stat;
    size_t total_write_size;
    size_t data_size = 0;
    char* data_ptr = nullptr;
    std::atomic<size_t> data_pos;

    size_t total_cuda_memory = 0;
    size_t total_cuda_free_memory = 0;
    vector<std::pair<size_t, size_t>> device_mem_info;

    phmap::flat_hash_map<string, std::pair<size_t, size_t>> dict;

    bool check_CUDA_device_props();
};
size_t calc_dltensor_size(const DLTensor* t);
void dltensor_deleter(DLManagedTensor* tensor);
}  // namespace icecake