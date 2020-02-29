#include "common.h"
#include <cstdlib>
#include "../include/icecake.hpp"

void dltensor_deleter_for_test(DLManagedTensor* tensor) {
    if (tensor->dl_tensor.shape) {
        free(tensor->dl_tensor.shape);
    }
    if (tensor->dl_tensor.strides) {
        free(tensor->dl_tensor.strides);
    }
    free(tensor->dl_tensor.data);
    free(tensor);
}
DLManagedTensor* make_DLM_tensor() {
    DLManagedTensor* dlm_tensor = (DLManagedTensor*) malloc(sizeof(DLManagedTensor));

    char* data = (char*) malloc(sizeof(char) * 100);
    ;
    dlm_tensor->dl_tensor.data = data;
    for (int i = 0; i < 100; i++) {
        data[i] = i % 255;
    }
    dlm_tensor->dl_tensor.byte_offset = 0;
    dlm_tensor->dl_tensor.ctx.device_id = 0;
    dlm_tensor->dl_tensor.ctx.device_type = DLDeviceType::kDLGPU;
    dlm_tensor->dl_tensor.dtype.bits = 32;
    dlm_tensor->dl_tensor.dtype.code = 2;
    dlm_tensor->dl_tensor.dtype.lanes = 1;
    dlm_tensor->dl_tensor.ndim = 3;
    dlm_tensor->dl_tensor.shape = (int64_t*) malloc(3 * sizeof(int64_t));
    dlm_tensor->dl_tensor.shape[0] = 1;
    dlm_tensor->dl_tensor.shape[0] = 4;
    dlm_tensor->dl_tensor.shape[0] = 25;
    dlm_tensor->dl_tensor.strides = NULL;
    dlm_tensor->deleter = dltensor_deleter_for_test;
    dlm_tensor->manager_ctx = NULL;

    return dlm_tensor;
}

int cmp_dlm_tensor(const DLManagedTensor* dlm_tensor1, const DLManagedTensor* dlm_tensor2) {
    if (dlm_tensor1 == nullptr) {
        return -1;
    }
    if (dlm_tensor2 == nullptr) {
        return 1;
    }
    auto tensor1 = dlm_tensor1->dl_tensor;
    auto tensor2 = dlm_tensor2->dl_tensor;

    size_t size1 = icecake::calc_dltensor_size(&tensor1);
    size_t size2 = icecake::calc_dltensor_size(&tensor2);
    if (size1 != size2){
        return size1 - size2;
    }
    return memcmp(tensor1.data, tensor2.data, size1);
}