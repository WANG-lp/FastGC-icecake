#include <gtest/gtest.h>  // googletest header file

#include "../include/icecake_export.hpp"

#include <vector>
#include <chrono>

using std::vector;

TEST(icecake, sub) {
    EXPECT_EQ(1, sub(1, 2));
    EXPECT_EQ(1, sub(2, 1));
}

TEST(icecake, sayhello) {
    EXPECT_EQ("hello,world,bob", csayhello("bob"));
}


TEST(cuda, sub) {
    EXPECT_EQ(1, sub_device(2, 1));
}

TEST(cuda, tensor){
    icecake::tensor tensor_;
    vector<int> vec = {1, 2, 3, 4, 5, 6};
    EXPECT_TRUE(tensor_.to_cuda_from_raw(0, vec.data(), sizeof(int) * vec.size()));
}
