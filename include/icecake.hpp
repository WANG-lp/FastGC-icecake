
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "dlpack.h"

using std::string;
using std::vector;

namespace pybind11 {
class capsule;
class array;
}  // namespace pybind11

namespace icecake {

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
    pybind11::array get_numpy_array(const string& fid);
    bool put_dltensor(const string& fid, pybind11::capsule capsule);
    pybind11::capsule get_dltensor(const string& fid, int device);

    void save_dltensor_to_file(const string& fid, const string& fname);
    void load_dltensor_from_file(const string& fname);

    size_t get_self_pointer_addr();
    void config_shuffle(bool enable_shuffle);
    void shuffle(size_t seed = 0);
    bool next_batch(size_t batch_size, vector<string>& names, bool auto_shuffle = true);

   private:
    std::mutex _lock;
    statistics stat;
    size_t total_write_size;
    size_t data_size = 0;
    char* data_ptr = nullptr;
    std::atomic<size_t> data_pos;

    size_t total_cuda_memory = 0;
    size_t total_cuda_free_memory = 0;
    vector<std::pair<size_t, size_t>> device_mem_info;
    std::unordered_map<string, std::pair<size_t, size_t>> dict;

    size_t seed = 0;
    bool enable_shuffle = true;
    vector<string> shuffled_array;
    std::atomic<size_t> shuffled_pos;

    bool check_CUDA_device_props();
};
size_t calc_dltensor_size(const DLTensor* t);
void dltensor_deleter(DLManagedTensor* tensor);
vector<char> serialize_dl_tensor(const DLTensor* t);
DLTensor deserialize_dl_tensor(const char* data);
}  // namespace icecake