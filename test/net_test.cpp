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
    constexpr int testNum = 1000;
    constexpr int testIter = 500;

    Random::Seed(42);
    IntNetwork* intNet = new IntNetwork();
    intNet->ResetWeight();

    for (int i = 0; i < iterNum; i++)
    {
        Train<IntNetwork>(*intNet, trainNum, 16, false);
    }
    double intDiffs[testNum];
    clock_t intDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        std::cout << i << "\r";
        intDuration += bitnet::Test<IntNetwork>(*intNet, testNum, 16, false, true, intDiffs);
    }

    std::cout << "\n\n\n";

    Random::Seed(42);
    BitNetwork *bitNet = new BitNetwork();
    bitNet->ResetWeight();

    for (int i = 0; i < iterNum; i++)
    {
        std::cout << i << "\r";
        Train<BitNetwork>(*bitNet, trainNum, 16, true);
    }
    double bitDiffs[testNum];
    clock_t bitDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        bitDuration += bitnet::Test<BitNetwork>(*bitNet, testNum, 16, true, true, bitDiffs);
    }

    std::cout << "int time: " << intDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit time: " << bitDuration / (double)CLOCKS_PER_SEC << std::endl;

    for (int i = 0; i < testNum; i++)
    {
        EXPECT_EQ(intDiffs[i], bitDiffs[i]);
    }
}