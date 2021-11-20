#ifndef TRAIN_H_
#define TRAIN_H_

#include "layers/layers.h"

namespace int_net
{
    using Input = IntInputLayer<2>;
    using Hidden1 = IntSignActivation<IntDenseLayer<Input, 32>>;
    using Hidden2 = IntSignActivation<IntDenseLayer<Hidden1, 16>>;
    using Output = IntDenseLayer<Hidden2, 1>;
    using Network = Output;
}

namespace bit_net
{
    using Input = BitInputLayer<2>;
    using Hidden1 = BitSignActivation<BitDenseLayer<Input, 32>>;
    using Hidden2 = BitSignActivation<BitDenseLayer<Hidden1, 16>>;
    using Output = BitDenseLayer<Hidden2, 1>;
    using Network = Output;
}

namespace train
{
    template <typename NetType>
    void Train(NetType &net, int nbTrain, double scale, bool shouldBitInput);

    template <typename NetType>
    void Test(NetType &net, int nbTest, double scale, bool shouldBitInput, double* diffOut);
}
#endif