#ifndef NET_COMMON_H_INCLUDED_
#define NET_COMMON_H_INCLUDED_

#include <cmath>
#include <cstdint>

namespace bitnet
{
	constexpr int BATCH_SIZE = 16;

	typedef double GradientType;
	typedef double BiasType;

	typedef double RealWeight;
	typedef int8_t IntBitWeight;
	typedef uint8_t BitWeight;

	typedef double RealType;
	typedef int IntType;
	typedef uint8_t BitBlock;
	typedef int8_t IntBitType;
}

#endif