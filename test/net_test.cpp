#include <gtest/gtest.h>

#include "../src/layers/layers.h"
#include "../src/net_common.h"
#include "../src/train.h"
#include <time.h>

TEST(BitNet, TrainSameCheck_Bit_Int)
{
    using namespace bitnet;
    constexpr int iterNum = 10;
    constexpr int trainNum = 1000;
    constexpr int testNum = 100000;

    Random::Seed(42);
    IntNetwork intNet;
    intNet.ResetWeight();

    for (int i = 0; i < iterNum; i++)
    {
        Train<IntNetwork>(intNet, trainNum, 16, false);
    }
    double intDiffs[testNum];
    clock_t start = clock();
    bitnet::Test<IntNetwork>(intNet, testNum, 16, false, true, intDiffs);
    clock_t stop = clock();
    int64_t intDuration = stop - start;

    std::cout << "\n\n\n";

    Random::Seed(42);
    BitNetwork bitNet;
    bitNet.ResetWeight();

    for (int i = 0; i < iterNum; i++)
    {
        Train<BitNetwork>(bitNet, trainNum, 16, true);
    }
    double bitDiffs[testNum];
    start = clock();
    bitnet::Test<BitNetwork>(bitNet, testNum, 16, true, true, bitDiffs);
    stop = clock();
    int64_t bitDuration = stop - start;

    for (int i = 0; i < testNum; i++)
    {
        EXPECT_EQ(intDiffs[i], bitDiffs[i]);
    }

    std::cout << "int time: " << intDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit time: " << bitDuration / (double)CLOCKS_PER_SEC << std::endl;
}