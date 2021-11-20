#include <iostream>

#include "train.h"
#include "util/make_data.h"
#include "net_common.h"

template <typename NetType>
double *Forward(NetType &net, const int8_t *input, const uint8_t *binInput);

template <>
double *Forward<int_net::Network>(int_net::Network &net, const int8_t *input, const uint8_t *binInput)
{
    return net.TrainForward(input);
}

template <>
double *Forward<bit_net::Network>(bit_net::Network &net, const int8_t *input, const uint8_t *binInput)
{
    return net.TrainForward(binInput);
}

template <typename NetType>
void Train(NetType &net, int nbTrain, double scale, bool shouldBitInput)
{
    constexpr int dataSize = 2;
    constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
    int8_t inputData[BATCH_SIZE * dataSize];
    int8_t teacherData[BATCH_SIZE];
    BitBlock binInput[BATCH_SIZE * padded_blocks];
    double diffs[BATCH_SIZE];
    double lr = 0.001;
    double maeSum = 0;
    for (int train = 0; train < nbTrain; train++)
    {
        util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);
        // util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);

        if (shouldBitInput)
        {
            util::BinarizeInputData(BATCH_SIZE, dataSize, inputData, binInput);
        }

        const double *pred = Forward<NetType>(net, inputData, binInput);

        double mae;
        double mse = util::CalcSquaredError(BATCH_SIZE, 1, scale, lr, pred, teacherData, diffs, &mae);

        net.TrainBackward(diffs);
        // std::cout << *pred << std::endl;
        maeSum += mae;
    }
    std::cout << maeSum / scale / nbTrain << std::endl;
}
template void Train<int_net::Network>(int_net::Network &net, int nbTrain, double scale, bool shouldBitInput);
template void Train<bit_net::Network>(bit_net::Network &net, int nbTrain, double scale, bool shouldBitInput);

template <typename NetType>
void Test(NetType &net, int nbTest, double scale)
{
    constexpr int dataSize = 2;
    constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
    int8_t inputData[BATCH_SIZE * 2];
    BitBlock binInput[BATCH_SIZE * padded_blocks];
    int8_t teacherData[BATCH_SIZE];
    for (int i = 0; i < nbTest; i++)
    {
        util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);
        // const double *pred = Forward<NetType>(net, inputData, binInput);
        const double *pred = Forward<NetType>(net, inputData, binInput);
        std::cout << (double)teacherData[0] << ":" << pred[0] << std::endl;
    }
}
template void Test<int_net::Network>(int_net::Network &net, int nbTest, double scale);
template void Test<bit_net::Network>(bit_net::Network &net, int nbTest, double scale);
