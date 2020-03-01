#include "../include/dlpack.h"
#include "../include/icecake.hpp"

#include "common.h"

int main(int argc, char** argv) {
    icecake::GPUCache gcache(4L * 1024 * 1024 * 1024);
    auto dlm_tensor = make_DLM_tensor();
    gcache.put_dltensor_to_device_memory("tensor1", dlm_tensor);

    auto dlm_tensor2 = gcache.get_dltensor_from_device("tensor1", 0);

    if (dlm_tensor2->deleter != NULL) {
        dlm_tensor2->deleter(dlm_tensor2);
    }
    return 0;
}