#include <gtest/gtest.h>

#include "../src/layers/layers.h"
#include "../src/net_common.h"
#include "../src/train.h"
#include <time.h>

// TEST(BitNet, TrainSameCheck_Bit_Int)
int main()
{
    using namespace bitnet;
    constexpr int iterNum = 10;
    constexpr int trainNum = 1000;
    constexpr int testNum = 10;
    constexpr int testIter = 500;

    Random::Seed(42);
    IntNetwork *intNet = reinterpret_cast<IntNetwork *>(_aligned_malloc(sizeof(IntNetwork), 32));
    intNet->ResetWeight();

    clock_t intTrainDuration = 0;
    for (int i = 0; i < iterNum; i++)
    {
        intTrainDuration += Train<IntNetwork>(*intNet, trainNum, 16, false);
    }
    GradientType intDiffs[testNum];
    clock_t intDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        std::cout << i << "\r";
        intDuration += bitnet::Test<IntNetwork>(*intNet, testNum, 16, false, true, intDiffs);
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
        bitTrainDuration += Train<BitNetwork>(*bitNet, trainNum, 16, true);
    }
    GradientType bitDiffs[testNum];
    clock_t bitDuration = 0;
    for (int i = 0; i < testIter; i++)
    {
        bitDuration += bitnet::Test<BitNetwork>(*bitNet, testNum, 16, true, true, bitDiffs);
    }

    std::cout << "int Train time: " << intTrainDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit Train time: " << bitTrainDuration / (double)CLOCKS_PER_SEC << std::endl
              << std::endl;

    std::cout << "int prediction time: " << intDuration / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "bit prediction time: " << bitDuration / (double)CLOCKS_PER_SEC << std::endl;

    // for (int i = 0; i < testNum; i++)
    // {
    //     EXPECT_EQ(intDiffs[i], bitDiffs[i]);
    // }
    return 0;
}