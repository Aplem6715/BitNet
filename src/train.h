#ifndef TRAIN_H_
#define TRAIN_H_

#include "layers/layers.h"
#include <time.h>

namespace bitnet
{
    using IInput = IntInputLayer<2>;
    using IHidden1 = IntSignActivation<IntDenseLayer<IInput, 32>>;
    using IHidden2 = IntSignActivation<IntDenseLayer<IHidden1, 16>>;
    using IOutput = IntDenseLayer<IHidden2, 1>;
    using IntNetwork = IOutput;

    using BInput = BitInputLayer<2>;
    using BHidden1 = BitSignActivation<BitDenseLayer<BInput, 32>>;
    using BHidden2 = BitSignActivation<BitDenseLayer<BHidden1, 16>>;
    using BOutput = BitDenseLayer<BHidden2, 1, true>;
    using BitNetwork = BOutput;

    template <typename NetType>
    void Train(NetType &net, int nbTrain, double scale, bool shouldBitInput);

    template <typename NetType>
    clock_t Test(NetType &net, int nbTest, double scale, bool shouldBitInput, bool isSilent, double *diffOut);
}
#endif
