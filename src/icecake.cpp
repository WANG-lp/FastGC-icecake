#include "../include/icecake_export.hpp"

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "cuda/cuda.cuh"
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
tensor::tensor() : data_(nullptr), data_size_(0), device_id_(-1), tensor_type_(-1){}
bool tensor::to_cuda(int device_id) {
    if (this->data_ == nullptr) {
        return false;
    }
    void *p = data_;
    size_t raw_len = data_size_;
    int t_type = tensor_type_;
    auto ret = to_cuda_from_raw(device_id, p, raw_len);
    // free memory
    if(t_type == 0){ // host
        free(p);
    }else if (t_type == 1){ // cuda device
        cudaFree(p);
    }
    return ret;
}
bool tensor::to_cuda_from_raw(int device_id, void* raw, size_t raw_len) {
    if(raw == nullptr){
        return false;
    }
    auto e = cudaSetDevice(device_id);
    if(e != cudaSuccess){
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

}