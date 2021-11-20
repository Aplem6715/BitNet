#include <gtest/gtest.h>

#include "../src/layers/layers.h"
#include "../src/net_common.h"
#include "../src/train.h"

TEST(BitNet, TrainSameCheck_Bit_Int)
{
    constexpr int iterNum = 10;
    constexpr int trainNum = 50;
    constexpr int testNum = 50;

    Random::Seed(42);
    int_net::Network intNet;
    intNet.ResetWeight();
    for (int i = 0; i < iterNum; i++)
    {
        train::Train<int_net::Network>(intNet, trainNum, 16, false);
    }
    double intDiffs[testNum];
    train::Test<int_net::Network>(intNet, testNum, 16, false, intDiffs);

    std::cout << "\n\n\n";

    Random::Seed(42);
    bit_net::Network bitNet;
    bitNet.ResetWeight();
    for (int i = 0; i < iterNum; i++)
    {
        train::Train<bit_net::Network>(bitNet, trainNum, 16, true);
    }
    double bitDiffs[testNum];
    train::Test<bit_net::Network>(bitNet, testNum, 16, true, bitDiffs);

    for (int i = 0; i < testNum; i++)
    {
        EXPECT_EQ(intDiffs[i], bitDiffs[i]);
    }
}