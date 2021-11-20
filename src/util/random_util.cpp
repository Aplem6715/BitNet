#include "random_util.h"

namespace Random
{

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> rnd_prob01(0.0, 1.0);

}