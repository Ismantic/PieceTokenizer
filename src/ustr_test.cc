#include "ustr.h"

#include "test.h"

#include <iostream>

namespace ustr {

TEST(UstrTest, EncodePODTest) {
    std::string tmp;
    {
        float v = 0.0;
        tmp = EncodePOD<float>(10.0);
        EXPECT_TRUE(DecodePOD<float>(tmp, &v));
        EXPECT_EQ(10.0, v);
    }

    {
        double v = 0.0;
        tmp = EncodePOD<double>(10.0);
        EXPECT_TRUE(DecodePOD<double>(tmp, &v));
        EXPECT_EQ(10.0, v);
    }

    {
        int32_t v = 0;
        tmp = ustr::EncodePOD<int32_t>(10);
        EXPECT_TRUE(DecodePOD<int32_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    {
        int16_t v = 0;
        tmp = EncodePOD<int16_t>(10);
        EXPECT_TRUE(DecodePOD<int16_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    {
        int64_t v = 0;
        tmp = ustr::EncodePOD<int64_t>(10);
        EXPECT_TRUE(DecodePOD<int64_t>(tmp, &v));
        EXPECT_EQ(10, v);
    }

    // Invalid data
    {
        int32_t v = 0;
        tmp = EncodePOD<int64_t>(10);
        EXPECT_FALSE(DecodePOD<int32_t>(tmp, &v));
    }
}


} // namespace ustr