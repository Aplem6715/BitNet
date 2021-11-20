/**
 * @file bit_helper.h
 * @author Daichi Sato
 * @brief SIMD演算ヘルパー
 * @version 1.0
 * @date 2021-11-20
 * 
 * @copyright Copyright (c) 2021 Daichi Sato
 * 
 */

#ifndef BIT_HELPER_H_
#define BIT_HELPER_H_

#include <intrin.h>
#include <cstdint>
#include <cmath>

namespace bitnet
{
    using vector32 = __m256i;

    constexpr int BYTE_BIT_WIDTH = 8;
    constexpr int SIMD_BIT_WIDTH = 256;

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

    /**
     * @brief 
     * -1/1を0/1で表したビット列の積和を計算する.
     * 入力ビット列はSIMD用にパディング済みである必要がある.
     * 
     * @param bitBlocks 
     * @param weightBlocks 
     * @param length ビット列の長さ
     * @return int 積和
     */
    inline int MaddPopcnt(const uint8_t *bitBlocks, const uint8_t *weightBlocks, const int length)
    {
        const int blocks = length / 256;
        vector32 x = _mm256_load_si256((vector32 *)bitBlocks);
        vector32 w = _mm256_load_si256((vector32 *)weightBlocks);
    }
}

#endif