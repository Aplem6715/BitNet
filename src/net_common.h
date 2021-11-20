#ifndef NET_COMMON_H_INCLUDED_
#define NET_COMMON_H_INCLUDED_

#include <cmath>
#include <cstdint>

constexpr int BATCH_SIZE = 16;
constexpr int BYTE_BIT_WIDTH = 8;
constexpr int SIMD_BIT_WIDTH = 256;

typedef double GradientType;
typedef double BiasType;

typedef double RealWeight;
typedef int8_t IntBitWeight;
typedef uint8_t BitWeight;

typedef double RealType;
typedef int IntType;
typedef uint8_t BitBlock;
typedef int8_t IntBitType;

constexpr int AddPaddingToBitSize(int bitSize)
{
	return std::ceil(bitSize / (double)SIMD_BIT_WIDTH) * SIMD_BIT_WIDTH;
}

constexpr int BitToBlockCount(int bitSize)
{
	return std::ceil(bitSize / (float)BYTE_BIT_WIDTH);
}

inline int GetBlockIndex(int bitIndex)
{
	return bitIndex / BYTE_BIT_WIDTH;
}

inline int GetBitIndexInBlock(int bitIndex)
{
	return BYTE_BIT_WIDTH - (bitIndex % BYTE_BIT_WIDTH) - 1;
}

#endif