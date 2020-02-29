#include "../include/icecake.hpp"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "cuda/cuda.cuh"
#include "utils/int2bytes.h"
using std::string;
using std::vector;
int sub(int a, int b) {
    if (a > b) {
        return a - b;
    } else {
        return b - a;
    }
}

std::string csayhello(const std::string& str) {
    vector<string> ss = {"hello,", "world,", str};
    string out;
    for (const auto s : ss) {
        out.append(s);
    }
    return out;
}

int sub_device(int a, int b) { return sub_cuda(a, b); }
namespace icecake {
tensor::tensor() : data_(nullptr), data_size_(0), device_id_(-1), tensor_type_(-1) {}
bool tensor::to_cuda(int device_id) {
    if (this->data_ == nullptr) {
        return false;
    }
    void* p = data_;
    size_t raw_len = data_size_;
    int t_type = tensor_type_;
    auto ret = to_cuda_from_raw(device_id, p, raw_len);
    // free memory
    if (t_type == 0) {  // host
        free(p);
    } else if (t_type == 1) {  // cuda device
        cudaFree(p);
    }
    return ret;
}
bool tensor::to_cuda_from_raw(int device_id, void* raw, size_t raw_len) {
    if (raw == nullptr) {
        return false;
    }
    auto e = cudaSetDevice(device_id);
    if (e != cudaSuccess) {
        spdlog::error("error while setting CUDA device {}", device_id);
        return false;
    }
    void* device_ptr_tmp = nullptr;
    e = cudaMalloc(&device_ptr_tmp, raw_len);
    if (e != cudaSuccess) {
        spdlog::error("error while malloc device memory on device {}", device_id);
        return false;
    }
    if (this->tensor_type_ == 0) {  // host -> device
        e = cudaMemcpy(device_ptr_tmp, raw, raw_len, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            spdlog::error("error while copy data from host to device {}", device_id);
            return false;
        }

    } else if (this->tensor_type_ == 1) {  // device -> device
        e = cudaMemcpy(device_ptr_tmp, raw, raw_len, cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) {
            spdlog::error("error while copy data from host to device {}", device_id);
            return false;
        }
    }
    // set internal variable
    data_ = device_ptr_tmp;
    data_size_ = raw_len;
    tensor_type_ = 1;
    device_id_ = device_id;
    return true;
}
tensor::~tensor() {
    if (this->data_ != nullptr) {
        if (this->tensor_type_ == 0) {  // host
            free(this->data_);
        } else if (this->tensor_type_ == 1) {  // cuda device
            cudaFree(this->data_);
        } else {
            spdlog::error("not implemented!");
        }
        this->data_ = nullptr;
        this->data_size_ = 0;
        spdlog::debug("deallocate success");
    }
}
}  // namespace icecake

namespace icecake {
size_t calc_dltensor_size(const DLTensor* t) {
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->shape[i];
    }
    size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    return size;
}
void dltensor_deleter(DLManagedTensor* tensor) {
    if (tensor->dl_tensor.shape) {
        free(tensor->dl_tensor.shape);
    }
    if (tensor->dl_tensor.strides) {
        free(tensor->dl_tensor.strides);
    }
    /****NOTICE***
    we do NOT need to delete underlying memory here, since we use a global memory
    block to store all tensor data in GPU
    //free(tensor->dl_tensor.data);
    ***END***/
    free(tensor);
}
inline vector<char> serialize_dl_tensor(const DLTensor* t) {
    vector<char> tmp_buff;
    size_t data_len = calc_dltensor_size(t);
    // DLDataType.code(uint8), DLDataType.bits(uint8), DLDataType.lanes(uint16), ndim(int),
    // shape[0](int64),shape[1]...shape[n], has_strides(char), strides[0](uint64),strides[1]...strides[n]
    size_t buff_len = sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint16_t) + sizeof(int) + t->ndim * sizeof(int64_t) +
                      sizeof(char) + data_len;
    if (t->strides != NULL) {
        buff_len += t->ndim * sizeof(int64_t);
    }
    tmp_buff.resize(buff_len);
    unsigned char* data = (unsigned char*) tmp_buff.data();
    data[0] = t->dtype.code;
    data += 1;
    data[0] = t->dtype.bits;
    data += 1;
    uint16_to_bytes(t->dtype.lanes, data);
    data += sizeof(uint16_t);
    uint32_to_bytes(t->ndim, data);
    data += sizeof(int);
    for (size_t i = 0; i < t->ndim; i++) {
        uint64_to_bytes(t->shape[i], data);
        data += sizeof(int64_t);
    }
    if (t->strides != NULL) {
        data[0] = 1;
        data += 1;
        for (size_t i = 0; i < t->ndim; i++) {
            uint64_to_bytes(t->strides[i], data);
            data += sizeof(int64_t);
        }
    } else {
        data[0] = 0;
        data += 1;
    }
    memcpy(data, (char*) t->data + t->byte_offset, data_len);

    return tmp_buff;
}

inline DLTensor deserialize_dl_tensor(const char* data) {
    if (data == nullptr) {
        spdlog::error("cannot deserialize nullptr to tensor");
        exit(1);
    }
    DLTensor t;
    size_t offset = 0;
    t.dtype.code = data[offset];
    offset += 1;
    t.dtype.bits = data[offset];
    offset += 1;
    t.dtype.lanes = bytes_to_uint16((unsigned char*) data + offset);
    offset += sizeof(uint16_t);
    t.ndim = bytes_to_uint32((unsigned char*) data + offset);
    offset += sizeof(int);
    t.shape = (int64_t*) malloc(sizeof(int64_t) * t.ndim);
    for (size_t i = 0; i < t.ndim; i++) {
        t.shape[i] = bytes_to_uint64((unsigned char*) data + offset);
        offset += sizeof(int64_t);
    }
    if (data[offset] != 0) {
        offset += 1;
        t.strides = (int64_t*) malloc(sizeof(int64_t) * t.ndim);
        for (size_t i = 0; i < t.ndim; i++) {
            t.strides[i] = bytes_to_uint64((unsigned char*) data + offset);
            offset += sizeof(int64_t);
        }
    } else {
        offset += 1;
        t.strides = NULL;
    }
    t.data = (void*) (data);
    t.byte_offset = 0;
    return t;
}
GPUCache::GPUCache(size_t alloc_size) : data_pos(0) {
    if (!check_CUDA_device_props()) {
        spdlog::error("Device does not meet the requirement");
        exit(1);
    }
    spdlog::info("Total CUDA memory size: {}, which is {:03.3f}GB", total_cuda_memory,
                 total_cuda_memory * 1.0f / (1024.0 * 1024 * 1024));
    spdlog::info("Total CUDA free memory size: {}, which is {:03.3f}GB", total_cuda_free_memory,
                 total_cuda_free_memory * 1.0f / (1024.0 * 1024 * 1024));
    if (alloc_size > total_cuda_free_memory) {
        spdlog::error("Requested cache size exceed free GPU memory size");
        exit(1);
    }
    if (cudaMallocManaged(&data_ptr, alloc_size, cudaMemAttachGlobal) != cudaSuccess) {
        spdlog::error("cannot alloc unified memory");
        exit(1);
    }
    data_size = alloc_size;
}
GPUCache::~GPUCache() {
    if (data_ptr != nullptr) {
        cudaFree(data_ptr);
    }
};

bool GPUCache::check_CUDA_device_props() {
    cudaGetDeviceCount(&(total_cuda_device));

    spdlog::info("There are {} CUDA devices", total_cuda_device);

    for (int cudaDevice = 0; cudaDevice < total_cuda_device; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (deviceProp.major < 6) {
            spdlog::error("device {} has compute capability {}.{}", deviceProp.name, deviceProp.major,
                          deviceProp.minor);
            return false;
        }
        if (deviceProp.concurrentManagedAccess == 0) {
            spdlog::error("device {} does not support concurrent Managed Access", deviceProp.name);
            return false;
        }
        size_t free_size, total_size;
        cudaSetDevice(cudaDevice);
        if (cudaMemGetInfo(&free_size, &total_size) != cudaSuccess) {
            spdlog::error("device {} get memory info failed", deviceProp.name);
        }
        total_cuda_memory += total_size;
        total_cuda_free_memory += free_size;
        device_mem_info.emplace_back(free_size, total_size);
    }
    cudaSetDevice(0);
    return true;
}
void GPUCache::write_to_device_memory(const string& fid, const char* blockRAW, size_t length, int device) {
    size_t old_size = data_pos.fetch_add(length);
    if (data_pos.load() > total_cuda_free_memory) {
        spdlog::error("exceed device free memory");
    }
    cudaMemcpy(data_ptr + old_size, blockRAW, length, cudaMemcpyDefault);
    cudaMemPrefetchAsync(data_ptr + old_size, length, device);  // migrating data to a GPU memory

    dict[fid] = {old_size, length};
}

char* GPUCache::read_from_device_memory(const string& fid, size_t* length) {
    auto pos_pair = get_pos(fid);
    if (pos_pair.first == 0 && pos_pair.second == 0) {
        *length = 0;
        return nullptr;
    }
    *length = pos_pair.second;
    return data_ptr + pos_pair.first;
}

char* GPUCache::read_from_device_memory(const string& fid) {
    size_t l;
    return read_from_device_memory(fid, &l);
}

bool GPUCache::put_dltensor_to_device_memory(const string& fid, DLManagedTensor* dltensor) {
    auto tmp_buff = serialize_dl_tensor(&(dltensor->dl_tensor));
    write_to_device_memory(fid, tmp_buff.data(), tmp_buff.size(), 0);
    if (dltensor->deleter != NULL) {
        dltensor->deleter(dltensor);
    }
    return true;
}

DLManagedTensor* GPUCache::get_dltensor_from_device(const string& fid, int device) {
    DLManagedTensor* dlm_tensor = (DLManagedTensor*) malloc(sizeof(DLManagedTensor));
    dlm_tensor->dl_tensor = deserialize_dl_tensor(read_from_device_memory(fid));
    dlm_tensor->dl_tensor.ctx.device_type = DLDeviceType::kDLGPU;
    dlm_tensor->dl_tensor.ctx.device_id = device;  // let cuda driver to copy memory
    dlm_tensor->manager_ctx = NULL;
    dlm_tensor->deleter = dltensor_deleter;
    return dlm_tensor;
}

std::pair<size_t, size_t> GPUCache::get_pos(const string& fid) {
    auto ret = dict.find(fid);
    if (ret != dict.end()) {
        return ret->second;
    }
    return {0, 0};
}

}  // namespace icecake