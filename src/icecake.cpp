

#include "../include/icecake.hpp"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <random>
#include <string>
#include <vector>
#include "cuda/cuda.cuh"
#include "utils/int2bytes.h"
namespace icecake {
inline bool dlpack_memory_row_major_test(DLTensor* dltensor, vector<size_t>& correct_strides) {
    auto shape = dltensor->shape;
    auto strides = dltensor->strides;
    int dim = dltensor->ndim;
    if (strides == nullptr || dim < 2) {
        return true;
    }
    for (int i = 0; i < correct_strides.size(); i++) {
        if (dltensor->strides[i] != correct_strides[i]) {
            return false;
        }
    }
    return true;
}
size_t calc_dltensor_size(const DLTensor* t) {
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->shape[i];
    }
    size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    return size;
}
void dltensor_deleter(DLManagedTensor* tensor) {
    /****NOTICE***
     * we do NOT need to delete underlying memory here, since we use a global device memory
     * block to store all tensor data in GPU
   // if (tensor->dl_tensor.shape) {
   //     free(tensor->dl_tensor.shape);
   // }
   // if (tensor->dl_tensor.strides) {
   //     free(tensor->dl_tensor.strides);
   // }
   //free(tensor->dl_tensor.data);
   ***END***/
    if (tensor) {
        free(tensor);
    }
}
vector<char> serialize_dl_tensor(const DLTensor* t) {
    vector<char> tmp_buff;
    // DLDataType.code(uint8), DLDataType.bits(uint8), DLDataType.lanes(uint16), ndim(int),
    // shape[0](int64),shape[1]...shape[n]
    size_t buff_len = sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint16_t) + sizeof(int) + t->ndim * sizeof(int64_t);

    tmp_buff.resize(buff_len);
    unsigned char* data = (unsigned char*) tmp_buff.data();
    data[0] = t->dtype.code;
    // spdlog::info("DLContext.DLDeviceType: {}", t->ctx.device_type);
    // spdlog::info("DLContext.device_id: {}", t->ctx.device_id);

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

    if (!(t->ctx.device_type == DLDeviceType::kDLGPU || t->ctx.device_type == DLDeviceType::kDLCPU)) {
        spdlog::error("Unsupported DLDeviceType {}", t->ctx.device_type);
        exit(1);
    }
    return tmp_buff;
}

DLTensor deserialize_dl_tensor(const char* data) {
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
    t.shape = (int64_t*) ((char*) data + offset);
    offset += sizeof(int64_t) * t.ndim;

    t.data = (void*) ((char*) data + offset);
    t.byte_offset = 0;
    t.strides = nullptr;
    return t;
}
GPUCache::GPUCache(size_t alloc_size) : data_pos(0) {
    if (!check_CUDA_device_props()) {
        spdlog::error("Device does not meet the requirement");
        exit(1);
    }
    stat.origin_size = 0;
    stat.total_read = 0;
    stat.total_write = 0;
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
void GPUCache::write_to_device_memory(const string& fid, const char* meta, size_t meta_length, const char* blockRAW,
                                      size_t length, int device) {
    size_t old_size = data_pos.fetch_add(meta_length + length);
    if (data_pos.load() > total_cuda_free_memory) {
        spdlog::error("exceed device free memory");
    }
    if (meta_length > 0) {
        cudaMemcpy(data_ptr + old_size, meta, meta_length, cudaMemcpyDefault);
    }
    cudaMemcpy(data_ptr + old_size + meta_length, blockRAW, length, cudaMemcpyDefault);
    cudaMemPrefetchAsync(data_ptr + old_size, meta_length + length, device);  // migrating data to a GPU memory
    _lock.lock();
    dict[fid] = {old_size, meta_length + length};
    _lock.unlock();
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
    size_t data_len = calc_dltensor_size(&(dltensor->dl_tensor));
    auto tmp_buff_head = serialize_dl_tensor(&(dltensor->dl_tensor));
    vector<size_t> strides_vec;
    strides_vec.resize(dltensor->dl_tensor.ndim);
    strides_vec[dltensor->dl_tensor.ndim - 1] = 1;
    if (dltensor->dl_tensor.ndim >= 2) {
        for (int i = dltensor->dl_tensor.ndim - 2; i >= 0; i--) {
            strides_vec[i] = dltensor->dl_tensor.shape[i + 1] * strides_vec[i + 1];
        }
    }
    if (dltensor->dl_tensor.strides == nullptr || dlpack_memory_row_major_test(&(dltensor->dl_tensor), strides_vec)) {
        write_to_device_memory(fid, tmp_buff_head.data(), tmp_buff_head.size(),
                               (const char*) (dltensor->dl_tensor.data), data_len, 0);
    } else {  // has stride, convert to row-majored (the last dimension is contiguous)
        spdlog::warn("tensor is not row-majored");
        vector<char> data_vec;
        data_vec.resize(data_len);
        const char* ori_data = (const char*) dltensor->dl_tensor.data;
#pragma omp parallel for
        for (size_t i = 0; i < data_len; i++) {
            size_t pos_in_dl_tensor = 0;
            size_t offset = i;
            for (int dim = 0; dim < dltensor->dl_tensor.ndim; dim++) {
                size_t n = offset / strides_vec[dim];
                pos_in_dl_tensor += n * dltensor->dl_tensor.strides[dim];
                offset -= n * strides_vec[dim];
            }
            data_vec[i] = ori_data[pos_in_dl_tensor + offset];
        }
        write_to_device_memory(fid, tmp_buff_head.data(), tmp_buff_head.size(), data_vec.data(), data_len, 0);
    }

    if (dltensor->deleter != nullptr) {
        dltensor->deleter(dltensor);
    }
    return true;
}

DLManagedTensor* GPUCache::get_dltensor_from_device(const string& fid, int device) {
    DLManagedTensor* dlm_tensor = (DLManagedTensor*) malloc(sizeof(DLManagedTensor));
    dlm_tensor->dl_tensor = deserialize_dl_tensor(read_from_device_memory(fid));
    dlm_tensor->dl_tensor.ctx.device_type = DLDeviceType::kDLGPU;
    // migrating data to device
    cudaMemPrefetchAsync(dlm_tensor->dl_tensor.data, calc_dltensor_size(&(dlm_tensor->dl_tensor)), device);
    dlm_tensor->dl_tensor.ctx.device_id = device;
    dlm_tensor->manager_ctx = this;
    dlm_tensor->deleter = dltensor_deleter;
    // dlm_tensor->manager_ctx = NULL;
    // dlm_tensor->deleter = NULL;
    return dlm_tensor;
}

void GPUCache::shuffle(size_t seed) {
    if (shuffled_array.size() != dict.size()) {  // update shuffled array
        shuffled_array.clear();
        _lock.lock();
        for (const auto& a : dict) {
            shuffled_array.push_back(a.first);
        }
        _lock.unlock();
    }
    if (enable_shuffle) {
        std::shuffle(shuffled_array.begin(), shuffled_array.end(), std::default_random_engine(seed));
    } else {
        std::sort(shuffled_array.begin(), shuffled_array.end());
    }
    this->seed = seed;
    shuffled_pos.store(0);
}
bool GPUCache::next_batch(size_t batch_size, vector<string>& names, bool auto_shuffle) {
    names.clear();
    size_t curr_pos = shuffled_pos.fetch_add(batch_size);
    if (curr_pos >= shuffled_array.size()) {
        if (auto_shuffle) {
            shuffle(this->seed + 1);
            return next_batch(batch_size, names, false);
        } else {
            return false;
        }
    }
    size_t end_pos = curr_pos + batch_size;
    if (end_pos > shuffled_array.size()) {
        for (; curr_pos < shuffled_array.size(); curr_pos++) {
            names.push_back(shuffled_array[curr_pos]);
        }
        for (; names.size() < batch_size;) {  // padding
            names.push_back(shuffled_array[shuffled_array.size() - 1]);
        }
    } else {
        for (; curr_pos < end_pos; curr_pos++) {
            names.push_back(shuffled_array[curr_pos]);
        }
    }
    return true;
}

std::pair<size_t, size_t> GPUCache::get_pos(const string& fid) {
    auto ret = dict.find(fid);
    if (ret != dict.end()) {
        return ret->second;
    }
    return {0, 0};
}

cv::Mat DatasetPipeline::Crop(const cv::Mat& img, int top, int left, int hight, int width) {
    cv::Rect roi;
    roi.x = top;
    roi.y = left;
    roi.width = left + width;
    roi.height = top + hight;

    /* Crop the original image to the defined ROI */

    cv::Mat crop = img(roi);
    return crop;
}

}  // namespace icecake
