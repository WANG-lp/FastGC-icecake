#include <gtest/gtest.h>  // googletest header file

#include "../include/icecake.hpp"

TEST(icecake, sub) {
    EXPECT_EQ(1, sub(1, 2));
    EXPECT_EQ(1, sub(2, 1));
}

TEST(icecake, sayhello) { EXPECT_EQ("hello,world,bob", csayhello("bob")); }
