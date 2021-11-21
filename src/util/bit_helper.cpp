#include "bit_helper.h"

namespace bitnet
{
    alignas(__m256i) static uint64_t TempMaddBuffer[4];
    int Popcnt2562(vector32 &vec)
    {
        int sum = 0;
        _mm256_store_si256((__m256i *)TempMaddBuffer, vec);
        // 256bitのpopcntはAVX2まででは存在しないので64bitずつカウント
        for (int i = 0; i < (SIMD_BIT_WIDTH / POPCNT_BIT_WIDTH); i++)
        {
            sum += _mm_popcnt_u64(TempMaddBuffer[i]);
        }
        return sum;
    }

    int MaddPopcnt2(const uint8_t *bitBlocks, const uint8_t *weightBlocks, const int length)
    {
        const int blocks = length / SIMD_BIT_WIDTH;
        int sum = 0;
        for (int b = 0; b < blocks; b++)
        {
            vector32 x = _mm256_load_si256((vector32 *)&(bitBlocks[b * SIMD_BYTE_WIDTH]));
            vector32 w = _mm256_load_si256((vector32 *)&(weightBlocks[b * SIMD_BYTE_WIDTH]));
            vector32 mul = ~_mm256_xor_si256(x, w);
            sum += Popcnt2562(mul);
        }
        return sum;
    }
}
