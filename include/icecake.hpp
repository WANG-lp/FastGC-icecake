#pragma once
#include <parallel_hashmap/phmap.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <string>
#include <vector>

using std::string;
using std::vector;

int sub(int a, int b);
std::string csayhello(const std::string& str);
int sub_device(int a, int b);

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

class GPUCache {
   public:
    int total_cuda_device = 0;

    GPUCache(size_t alloc_size);
    ~GPUCache();

    void write_to_device_memory(const string& fid, const char* blockRAW, size_t length, int device);
    char* read_from_device_memory(const string& fid, size_t *length);
    char* read_from_device_memory(const string& fid);
    std::pair<size_t, size_t> get_pos(const string& fid);

   private:
    size_t data_size = 0;
    char* data_ptr = nullptr;
    std::atomic<size_t> data_pos;

    size_t total_cuda_memory = 0;
    size_t total_cuda_free_memory = 0;
    vector<std::pair<size_t, size_t>> device_mem_info;

    phmap::flat_hash_map<string, std::pair<size_t, size_t>> dict;

    bool check_CUDA_device_props();
};
}  // namespace icecake