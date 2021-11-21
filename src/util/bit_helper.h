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
    constexpr int POPCNT_BIT_WIDTH = 64;
    constexpr int SIMD_BIT_WIDTH = 256;
    constexpr int SIMD_BYTE_WIDTH = SIMD_BIT_WIDTH / BYTE_BIT_WIDTH;

    constexpr int AddPaddingToBytes(int byteSize)
    {
        return std::ceil(byteSize / (double)(SIMD_BYTE_WIDTH)) * SIMD_BYTE_WIDTH;
    }

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

    alignas(__m256i) static uint64_t TempMaddBuffer[4];
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
        const int blocks = length / SIMD_BIT_WIDTH;
        int sum = 0;
        for (int b = 0; b < blocks; b++)
        {
            const int blockShift = b * SIMD_BYTE_WIDTH;
            vector32 x = _mm256_load_si256((vector32 *)&(bitBlocks[blockShift]));
            vector32 w = _mm256_load_si256((vector32 *)&(weightBlocks[blockShift]));
            vector32 mul = ~_mm256_xor_si256(x, w);

            _mm256_store_si256((__m256i *)TempMaddBuffer, mul);
            // 256bitのpopcntはAVX2まででは存在しないので64bitずつカウント
            for (int i = 0; i < (SIMD_BIT_WIDTH / POPCNT_BIT_WIDTH); i++)
            {
                sum += __popcnt64(TempMaddBuffer[i]);
            }
        }
        return sum;
    }

    /**
     * @brief 各バイトの符号（MSB）を抽出してビット列を作成する
     * 
     * @param inputs 入力バイト列
     * @param dst 生成されるビット列の格納先. 長さ[byteLength÷32]の配列アドレス
     * @param byteLength 入力バイト数
     */
    inline void CollectSignBit(const int8_t *inputs, int *dst, const int byteLength)
    {
        const int blocks = byteLength / SIMD_BYTE_WIDTH;
        for (int b = 0; b < blocks; b++)
        {
            const int blockShift = b * SIMD_BYTE_WIDTH;
            vector32 x = _mm256_load_si256((vector32 *)&(inputs[blockShift]));
            dst[b] = _mm256_movemask_epi8(x);
        }
    }
}

#endif