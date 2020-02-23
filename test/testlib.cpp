#include <gtest/gtest.h>  // googletest header file

#include "../include/icecake.hpp"

#include <chrono>
#include <string>
#include <vector>

using std::string;
using std::vector;

TEST(icecake, sub) {
    EXPECT_EQ(1, sub(1, 2));
    EXPECT_EQ(1, sub(2, 1));
}

TEST(icecake, sayhello) { EXPECT_EQ("hello,world,bob", csayhello("bob")); }

TEST(cuda, sub) { EXPECT_EQ(1, sub_device(2, 1)); }

TEST(cuda, tensor) {
    icecake::tensor tensor_;
    vector<int> vec = {1, 2, 3, 4, 5, 6};
    EXPECT_TRUE(tensor_.to_cuda_from_raw(0, vec.data(), sizeof(int) * vec.size()));
}

TEST(GPUCache, write_to_device_memory) {
    icecake::GPUCache gcache(60L * 1024 * 1024 * 1024);
    string str = "hello,world!";
    string str2 = "little dog";
    gcache.write_to_device_memory("block1", str.data(), str.size(), 0);
    if (gcache.total_cuda_device > 1) {
        gcache.write_to_device_memory("block2", str2.data(), str2.size(), 1);
    }
    EXPECT_EQ(memcmp(gcache.read_from_device_memory("block1"), str.data(), str.size()), 0);
    EXPECT_EQ(memcmp(gcache.read_from_device_memory("block2"), str2.data(), str2.size()), 0);
}