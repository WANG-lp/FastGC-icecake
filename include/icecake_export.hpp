#pragma once
#include <spdlog/spdlog.h>
#include <string>

int sub(int a, int b);
std::string csayhello(const std::string& str);
int sub_device(int a, int b);

namespace icecake {
class tensor {
   public:
    tensor();
    ~tensor();
    bool to_cuda(int device_id = 0);
    bool to_cuda_from_raw(int device_id = 0, void *raw = nullptr, size_t raw_len = 0);


    // ---------data area--------------
    int tensor_type_;  // 0->host, 1->cuda
    int device_id_;
    void* data_;  // data_pointer;
    size_t data_size_;
};
}