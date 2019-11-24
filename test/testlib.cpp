#include <gtest/gtest.h>  // googletest header file

#include "../include/lib.h"

TEST(Lib, sub) {
  EXPECT_EQ(1, sub(1, 2));
  EXPECT_EQ(1, sub(2, 1));
}
