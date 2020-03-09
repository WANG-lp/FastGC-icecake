#include <gtest/gtest.h>  // googletest header file

#include "../include/icecake.hpp"
#include "common.h"

#include <chrono>
#include <string>
#include <vector>

using std::string;
using std::vector;

TEST(GPUCache, write_to_device_memory) {
    icecake::GPUCache gcache(60L * 1024 * 1024 * 1024);
    string str = "hello,world!";
    string str2 = "little dog";
    gcache.write_to_device_memory("block1", nullptr, 0, str.data(), str.size(), 0);
    if (gcache.total_cuda_device > 1) {
        gcache.write_to_device_memory("block2", nullptr, 0, str2.data(), str2.size(), 1);
    }
    EXPECT_EQ(memcmp(gcache.read_from_device_memory("block1"), str.data(), str.size()), 0);
    EXPECT_EQ(memcmp(gcache.read_from_device_memory("block2"), str2.data(), str2.size()), 0);
}

TEST(GPUCache, DLM_Tensor) {
    icecake::GPUCache gcache(4L * 1024 * 1024 * 1024);
    auto dlm_tensor = make_DLM_tensor();
    dlm_tensor->deleter = NULL;

    gcache.put_dltensor_to_device_memory("tensor1", dlm_tensor);

    auto dlm_tensor2 = gcache.get_dltensor_from_device("tensor1", 0);

    EXPECT_EQ(cmp_dlm_tensor(dlm_tensor, dlm_tensor2), 0);

    if (dlm_tensor2->deleter != nullptr) {
        dlm_tensor2->deleter(dlm_tensor2);
    }
    dltensor_deleter_for_test(dlm_tensor);
}