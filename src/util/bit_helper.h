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
    using vector16 = __m128i;
    using float8 = __m256;
    using float4 = __m128;

    constexpr int BYTE_BIT_WIDTH = 8;
    constexpr int INT32_BIT_WIDTH = 32;
    constexpr int FLOAT_BIT_WIDTH = 32;
    constexpr int POPCNT_BIT_WIDTH = 64;

    constexpr int SIMD_BIT_WIDTH = 256;

    constexpr int NUM_BYTES_IN_REGISTER = SIMD_BIT_WIDTH / BYTE_BIT_WIDTH;
    constexpr int NUM_FLOAT_IN_REGISTER = SIMD_BIT_WIDTH / FLOAT_BIT_WIDTH;

    constexpr int NUM_BYTES_IN_FLOATS = FLOAT_BIT_WIDTH / BYTE_BIT_WIDTH;

    constexpr int AddPaddingToBytes(int byteSize)
    {
        return std::ceil(byteSize / (double)(NUM_BYTES_IN_REGISTER)) * NUM_BYTES_IN_REGISTER;
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
        return bitIndex % BYTE_BIT_WIDTH;
    }

    inline double sgn(double val)
    {
        return (double(0) < val) - (val < double(0));
    }

    static const unsigned char BitReverseTable[] =
        {
            0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
            0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
            0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
            0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
            0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
            0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
            0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
            0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
            0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
            0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
            0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
            0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
            0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
            0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
            0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
            0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

    namespace
    {
        // 遅い上にSegment faultするから廃止
        // static const uint8_t mask1a[32] = {
        //     0x00, 0x00, 0x00, 0x00, //
        //     0x00, 0x00, 0x00, 0x00, //
        //     0x01, 0x01, 0x01, 0x01, //
        //     0x01, 0x01, 0x01, 0x01, //
        //     0x02, 0x02, 0x02, 0x02, //
        //     0x02, 0x02, 0x02, 0x02, //
        //     0x03, 0x03, 0x03, 0x03, //
        //     0x03, 0x03, 0x03, 0x03  //
        // };

        // static const uint8_t mask2a[32] = {
        //     0x01, 0x02, 0x04, 0x08, //
        //     0x10, 0x20, 0x40, 0x80, //
        //     0x01, 0x02, 0x04, 0x08, //
        //     0x10, 0x20, 0x40, 0x80, //
        //     0x01, 0x02, 0x04, 0x08, //
        //     0x10, 0x20, 0x40, 0x80, //
        //     0x01, 0x02, 0x04, 0x08, //
        //     0x10, 0x20, 0x40, 0x80, //
        // };

        // inline void BroadcastBitsToBytes32(int bits, vector32 *target)
        // {
        //     vector32 mask2 = _mm256_load_si256((vector32 *)mask2a);
        //     vector32 mask1 = _mm256_load_si256((vector32 *)mask1a);

        //     vector32 y = _mm256_set1_epi32(bits);
        //     vector32 z = _mm256_shuffle_epi8(y, mask1);
        //     *target = _mm256_and_si256(z, mask2);
        // }

        // inline void SetSignBit(const __m256 &origin, const vector32 &expandSigns, float8 *result)
        // {
        //     __m256 msb_mask = _mm256_set1_ps(-0.0);

        //     float8 cvt = _mm256_cvtepi32_ps(expandSigns);
        //     float8 maskedSigns = _mm256_and_ps(cvt, msb_mask);
        //     *result = _mm256_xor_ps(origin, maskedSigns);
        // }

        // inline void AddFloat8(float *target, float8 addition)
        // {
        //     float8 target8 = _mm256_load_ps(target);
        //     target8 = _mm256_add_ps(target8, addition);
        //     _mm256_store_ps(target, target8);
        // }

        // inline void NegateAddFloats(float *floats, float diff, const uint8_t *bits, const int len)
        // {
        //     // SIMD演算ループ回数
        //     const int xBlocks = len / (NUM_FLOAT_IN_REGISTER * NUM_BYTES_IN_FLOATS);
        //     // 未アライン領域数　端数
        //     const int xOver = len % (NUM_FLOAT_IN_REGISTER * NUM_BYTES_IN_FLOATS);

        //     const int *bits32 = reinterpret_cast<const int *>(bits);
        //     float *f_cur = floats;
        //     float8 diff8 = _mm256_broadcast_ss(&diff);

        //     int intBlock;
        //     for (intBlock = 0; intBlock < xBlocks; intBlock++)
        //     {
        //         vector32 zeros32 = _mm256_setzero_si256();
        //         vector32 signs32;
        //         BroadcastBitsToBytes32(*bits32, &signs32);

        //         // 0の位置が1になるマスク
        //         signs32 = _mm256_cmpeq_epi8(zeros32, signs32);
        //         vector16 low16 = _mm256_extracti128_si256(signs32, 0);
        //         vector16 high16 = _mm256_extracti128_si256(signs32, 1);

        //         // signs8x4[0]
        //         float8 signedDiff;
        //         vector32 expandSigns = _mm256_cvtepi8_epi32(low16);
        //         SetSignBit(diff8, expandSigns, &signedDiff);
        //         AddFloat8(f_cur, signedDiff);
        //         f_cur += NUM_FLOAT_IN_REGISTER;

        //         // signs8x4[1]
        //         expandSigns = _mm256_cvtepi8_epi32(_mm_srli_si128(low16, 8));
        //         SetSignBit(diff8, expandSigns, &signedDiff);
        //         AddFloat8(f_cur, signedDiff);
        //         f_cur += NUM_FLOAT_IN_REGISTER;

        //         // signs8x4[2]
        //         expandSigns = _mm256_cvtepi8_epi32(high16);
        //         SetSignBit(diff8, expandSigns, &signedDiff);
        //         AddFloat8(f_cur, signedDiff);
        //         f_cur += NUM_FLOAT_IN_REGISTER;

        //         // signs8x4[3]
        //         expandSigns = _mm256_cvtepi8_epi32(_mm_srli_si128(high16, 8));
        //         SetSignBit(diff8, expandSigns, &signedDiff);
        //         AddFloat8(f_cur, signedDiff);
        //         f_cur += NUM_FLOAT_IN_REGISTER;

        //         ++bits32;
        //     }

        //     int overStarts = intBlock * (INT32_BIT_WIDTH / BYTE_BIT_WIDTH);
        //     for (int i = 0; i < xOver; i++)
        //     {
        //         int blockIdx = GetBlockIndex(overStarts + i);
        //         int bitShift = GetBitIndexInBlock(overStarts + i);
        //         if ((bits[blockIdx] >> bitShift) & 1)
        //         {
        //             *f_cur += diff;
        //         }
        //         else
        //         {
        //             *f_cur -= diff;
        //         }
        //         ++f_cur;
        //     }
        // }

        // // 参考：https://qiita.com/beru/items/fff00c19968685dada68
        // inline __m128 hsum128_ps(__m128 x)
        // {
        //     // loDual = ( -, -, x1, x0 )
        //     const __m128 loDual = x;
        //     // hiDual = ( -, -, x3, x2 )
        //     const __m128 hiDual = _mm_movehl_ps(x, x);
        //     // sumDual = ( -, -, x1+x3, x0+x2 )
        //     const __m128 sumDual = _mm_add_ps(loDual, hiDual);
        //     // lo = ( -, -, -, x0+x2 )
        //     const __m128 lo = sumDual;
        //     // hi = ( -, -, -, x1+x3 )
        //     const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
        //     // sum = ( -, -, -, x0+x1+x2+x3 )
        //     const __m128 sum = _mm_add_ss(lo, hi);
        //     return sum;
        // }

        // // 参考：https://qiita.com/beru/items/fff00c19968685dada68
        // inline float hsum4x256_ps(const __m256 &a,
        //                           const __m256 &b,
        //                           const __m256 &c,
        //                           const __m256 &d)
        // {
        //     __m256 t0, t1, t2, t3;
        //     __m256 tt0, tt1;
        //     t0 = _mm256_unpacklo_ps(a, b); // b5,a5,b4,a4, b1,a1,b0,a0
        //     t1 = _mm256_unpackhi_ps(a, b); // b7,a7,b6,a6, b3,a3,b2,a2
        //     t2 = _mm256_unpacklo_ps(c, d); // d5,c5,d4,c4, d1,c1,d0,c0
        //     t3 = _mm256_unpackhi_ps(c, d); // d7,c7,d6,c6, d3,c3,d2,c2

        //     tt0 = _mm256_add_ps(t0, t1); // b57,a57,b46,a46, b13,a13,b02,a02
        //     tt1 = _mm256_add_ps(t2, t3); // d57,c57,d46,c46, d13,c13,d02,c02

        //     t0 = _mm256_shuffle_ps(tt0, tt1, _MM_SHUFFLE(1, 0, 1, 0)); // d46,c46,b46,a46, d02,c02,b02,a02
        //     t1 = _mm256_shuffle_ps(tt0, tt1, _MM_SHUFFLE(3, 2, 3, 2)); // d57,c57,b57,a57, d13,c13,b13,a13

        //     tt0 = _mm256_add_ps(t0, t1); // d4567,c4567,b4567,a4567, d0123,c0123,b0123,a0123
        //     __m128 upper = _mm256_extractf128_ps(tt0, 1);

        //     return _mm_cvtss_f32(hsum128_ps(_mm_add_ps(_mm256_castps256_ps128(tt0), upper)));
        // }
        /**
         * @brief -1/1を表すビット列とfloat列の積を計算し総和を求める
         *
         * @param bits ビット列
         * @param f_input float列
         * @param len float列の長さ
         * @return float 積和
         */
        // inline float MaddReal(const int *bits, const float *f_input, const int len)
        // {
        //     // SIMD演算ループ回数
        //     const int xBlocks = len / NUM_FLOAT_IN_REGISTER;
        //     // 未アライン領域数　端数
        //     const int xOver = len % NUM_FLOAT_IN_REGISTER;

        //     float sum = 0;

        //     int intBlock;
        //     for (intBlock = 0; intBlock < xBlocks; intBlock++)
        //     {
        //         const int intShift = intBlock * NUM_BYTES_IN_FLOATS;

        //         const __m256 msb_mask = _mm256_setzero_ps();
        //         vector32 zeros32 = _mm256_setzero_si256();
        //         vector32 signs32 = BroadcastBitsToBytes32(bits[intShift]);
        //         // 0の位置が1になるマスク
        //         signs32 = _mm256_cmpeq_epi64(zeros32, signs32);
        //         vector16 low16 = _mm256_extracti128_si256(signs32, 0);
        //         vector16 high16 = _mm256_extracti128_si256(signs32, 0);

        //         // signs8x4[0]
        //         vector32 expandSigns = _mm256_cvtepu8_epi32(low16);
        //         float8 maskedSigns = _mm256_and_ps(_mm256_cvtepi32_ps(expandSigns), msb_mask);
        //         float8 x = _mm256_load_ps(f_input);
        //         float8 mul0 = _mm256_xor_ps(x, maskedSigns);

        //         // signs8x4[1]
        //         const int shift_f_register1 = NUM_FLOAT_IN_REGISTER;
        //         expandSigns = _mm256_cvtepu8_epi32(_mm_srli_si128(low16, 8));
        //         maskedSigns = _mm256_and_ps(_mm256_cvtepi32_ps(expandSigns), msb_mask);
        //         x = _mm256_load_ps(&(f_input[shift_f_register1]));
        //         float8 mul1 = _mm256_xor_ps(x, maskedSigns);

        //         // signs8x4[2]
        //         const int shift_f_register2 = NUM_FLOAT_IN_REGISTER * 2;
        //         expandSigns = _mm256_cvtepu8_epi32(high16);
        //         maskedSigns = _mm256_and_ps(_mm256_cvtepi32_ps(expandSigns), msb_mask);
        //         x = _mm256_load_ps(&(f_input[shift_f_register2]));
        //         float8 mul2 = _mm256_xor_ps(x, maskedSigns);

        //         // signs8x4[3]
        //         const int shift_f_register3 = NUM_FLOAT_IN_REGISTER * 3;
        //         expandSigns = _mm256_cvtepu8_epi32(_mm_srli_si128(high16, 8));
        //         maskedSigns = _mm256_and_ps(_mm256_cvtepi32_ps(expandSigns), msb_mask);
        //         x = _mm256_load_ps(&(f_input[shift_f_register3]));
        //         float8 mul3 = _mm256_xor_ps(x, maskedSigns);

        //         sum += hsum4x256_ps(mul0, mul1, mul2, mul3);
        //     }

        //     for (int i = 0; i < xOver; i++)
        //     {
        //         // const int bitShift = GetBitIndexInBlock(i);
        //         // const BitBlock b = bits[intBlock * NUM_BYTES_IN_FLOATS + i];
        //         // const BitBlock w_bit = (b >> bitShift) & 0b1;
        //         // const int weight = (w_bit == 1 ? 1 : -1);
        //         // sum += nextGrad[batchShiftOut + i_out] * weight;
        //     }
        // }

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
    inline int MaddPopcnt2(const uint8_t *bitBlocks, const uint8_t *weightBlocks, const int length)
    {
        alignas(__m256i) static uint64_t TempMaddBuffer[4];
        const int blocks = length / SIMD_BIT_WIDTH;
        int sum = 0;
        for (int b = 0; b < blocks; b++)
        {
            vector32 x = _mm256_load_si256((vector32 *)&(bitBlocks[b * NUM_BYTES_IN_REGISTER]));
            vector32 w = _mm256_load_si256((vector32 *)&(weightBlocks[b * NUM_BYTES_IN_REGISTER]));
            vector32 mul = ~_mm256_xor_si256(x, w);

            _mm256_store_si256((__m256i *)TempMaddBuffer, mul);
            // 256bitのpopcntはAVX2まででは存在しないので64bitずつカウント
            for (int i = 0; i < (SIMD_BIT_WIDTH / POPCNT_BIT_WIDTH); i++)
            {
                sum += _mm_popcnt_u64(TempMaddBuffer[i]);
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
        const int blocks = byteLength / NUM_BYTES_IN_REGISTER;
        for (int b = 0; b < blocks; b++)
        {
            const int blockShift = b * NUM_BYTES_IN_REGISTER;
            vector32 x = _mm256_load_si256((vector32 *)(inputs + blockShift));

            // 要素が0のバイトの最上位ビットが1になるよう調整
            vector32 zero = _mm256_setzero_si256();
            x = _mm256_or_si256(x, _mm256_cmpeq_epi8(x, zero));

            dst[b] = _mm256_movemask_epi8(~x);
        }
    }
}

#endif