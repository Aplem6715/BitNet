#include <iostream>

#include "train.h"
#include "util/make_data.h"
#include "net_common.h"

namespace bitnet
{
    template <typename NetType>
    int32_t *TrainForward(NetType &net, const int8_t *input, const uint8_t *binInput);

    template <>
    int32_t *TrainForward<IntNetwork>(IntNetwork &net, const int8_t *input, const uint8_t *binInput)
    {
        return net.TrainForward(input);
    }

    template <>
    int32_t *TrainForward<BitNetwork>(BitNetwork &net, const int8_t *input, const uint8_t *binInput)
    {
        return net.TrainForward(binInput);
    }

    template <typename NetType>
    const int32_t *Forward(NetType &net, const int8_t *input, const uint8_t *binInput);

    template <>
    const int32_t *Forward<IntNetwork>(IntNetwork &net, const int8_t *input, const uint8_t *binInput)
    {
        return net.Forward(input);
    }

    template <>
    const int32_t *Forward<BitNetwork>(BitNetwork &net, const int8_t *input, const uint8_t *binInput)
    {
        return net.Forward(binInput);
    }

    template <typename NetType>
    void Train(NetType &net, int nbTrain, double scale, bool shouldBitInput)
    {
        constexpr int dataSize = 2;
        constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
        int8_t inputData[BATCH_SIZE * dataSize];
        int8_t teacherData[BATCH_SIZE];
        BitBlock binInput[BATCH_SIZE * padded_blocks];
        GradientType diffs[BATCH_SIZE];
        double lr = 0.0001;
        double maeSum = 0;
        for (int train = 0; train < nbTrain; train++)
        {
            util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);
            // util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);

            if (shouldBitInput)
            {
                util::BinarizeInputData(BATCH_SIZE, dataSize, inputData, binInput);
            }

            const int32_t *pred = TrainForward<NetType>(net, inputData, binInput);

            double mae;
            double mse = util::CalcSquaredError(BATCH_SIZE, 1, scale, lr, pred, teacherData, diffs, &mae);

            net.TrainBackward(diffs);
            // std::cout << *pred << std::endl;
            maeSum += mae;
        }
        std::cout << maeSum / scale / nbTrain << std::endl;
    }
    template void Train<IntNetwork>(IntNetwork &net, int nbTrain, double scale, bool shouldBitInput);
    template void Train<BitNetwork>(BitNetwork &net, int nbTrain, double scale, bool shouldBitInput);

    template <typename NetType>
    clock_t Test(NetType &net, int nbTest, double scale, bool shouldBitInput, bool isSilent, float *diffOut)
    {
        constexpr int dataSize = 2;
        constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
        int8_t inputData[BATCH_SIZE * 2];
        BitBlock binInput[BATCH_SIZE * padded_blocks];
        int8_t teacherData[BATCH_SIZE];
        clock_t timer = 0;
        for (int i = 0; i < nbTest; i++)
        {
            util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);

            if (shouldBitInput)
            {
                util::BinarizeInputData(BATCH_SIZE, dataSize, inputData, binInput);
            }

            clock_t start = clock();
            const int32_t *pred = Forward<NetType>(net, inputData, binInput);
            clock_t stop = clock();
            timer += stop - start;

            diffOut[i] = (float)teacherData[0] - pred[0];
            if (!isSilent)
            {
                std::cout << (float)teacherData[0] << ":" << pred[0] << std::endl;
            }
        }
        return timer;
    }
    template clock_t Test<IntNetwork>(IntNetwork &net, int nbTest, double scale, bool shouldBitInput, bool isSilent, float *diffOut);
    template clock_t Test<BitNetwork>(BitNetwork &net, int nbTest, double scale, bool shouldBitInput, bool isSilent, float *diffOut);
}
