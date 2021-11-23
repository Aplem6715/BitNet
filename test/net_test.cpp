#include <gtest/gtest.h>

#include "../src/layers/layers.h"
#include "../src/net_common.h"
#include "../src/train.h"
#include <time.h>

// 256-256 SIMD grad
// int Train time: 88.557
// bit Train time: 112.216
// int prediction time: 86
// bit prediction time: 6.602

// int Train time: 88.602
// bit Train time: 112.192

// int prediction time: 86.385
// bit prediction time: 6.536

// 256-256 normal
// int Train time: 89.735
// bit Train time: 120.585
// int prediction time: 85.458
// bit prediction time: 6.499

TEST(BitNet, TrainSameCheck_Bit_Int)
// int main()
{
    using namespace bitnet;
    constexpr int iterNum = 10;
    constexpr int trainNum = 1000;
    constexpr int testNum = 10;
    constexpr int testIter = 50000;
    float scale = 256;

    Random::Seed(42);
    IntNetwork *intNet = reinterpret_cast<IntNetwork *>(_aligned_malloc(sizeof(IntNetwork), 32));
    intNet->ResetWeight();

    clock_t intTrainDuration = 0;
    for (int i = 0; i < iterNum; i++)
    {
        intTrainDuration += Train<IntNetwork>(*intNet, trainNum, scale, false);
    }
    GradientType intDiffs[testNum];
    clock_t intDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        std::cout << i << "\r";
        intDuration += bitnet::Test<IntNetwork>(*intNet, testNum, scale, false, true, intDiffs);
    }

    std::cout << "\n\n\n";

    Random::Seed(42);
    BitNetwork *bitNet = reinterpret_cast<BitNetwork *>(_aligned_malloc(sizeof(BitNetwork), 32));
    bitNet->Init();
    bitNet->ResetWeight();

    clock_t bitTrainDuration = 0;
    for (int i = 0; i < iterNum; i++)
    {
        std::cout << i << "\r";
        bitTrainDuration += Train<BitNetwork>(*bitNet, trainNum, scale, true);
    }
    GradientType bitDiffs[testNum];
    clock_t bitDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        bitDuration += bitnet::Test<BitNetwork>(*bitNet, testNum, scale, true, true, bitDiffs);
    }

    std::cout << "int Train time: " << intTrainDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit Train time: " << bitTrainDuration / (double)CLOCKS_PER_SEC << std::endl
              << std::endl;

    std::cout << "int prediction time: " << intDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit prediction time: " << bitDuration / (double)CLOCKS_PER_SEC << std::endl;

    for (int i = 0; i < testNum; i++)
    {
        EXPECT_EQ(intDiffs[i], bitDiffs[i]);
    }
    // return 0;
}