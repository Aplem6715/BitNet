/**
 * @file bit_input.h
 * @author Daichi Sato
 * @brief bitニューラルネット用の入力層
 * @version 1.0
 * @date 2021-11-20
 * 
 * @copyright Copyright (c) 2021 Daichi Sato
 * 
 * 必要に応じて入力にパディングビットを追加する。
 * 
 */
#ifndef BIT_INPUT_H_
#define BIT_INPUT_H_

#include "../../net_common.h"
#include "../../util/bit_helper.h"

namespace bitnet
{
    /**
     * @brief Int型入力層
     * 
     * @tparam PreviousLayer_t 前のレイヤー型
     * @tparam OutputBits 入力数
     */
    template <int InputBits>
    class BitInputLayer
    {
    public:
        // 出力次元数
        static constexpr int COMPRESS_OUT_DIM = InputBits;
        static constexpr int COMPRESS_OUT_BITS = InputBits;
        static constexpr int COMPRESS_OUT_BLOCKS = BitToBlockCount(InputBits);
        static constexpr int PADDED_OUT_BITS = AddPaddingToBitSize(COMPRESS_OUT_BITS);
        static constexpr int PADDED_OUT_BLOCKS = BitToBlockCount(PADDED_OUT_BITS);

    private:
        // 出力バッファ（次の層が参照する
        alignas(32) BitBlock _outputBuffer[PADDED_OUT_BLOCKS] = {};
        alignas(32) BitBlock _outputBatchBuffer[BATCH_SIZE * PADDED_OUT_BLOCKS] = {};

    public:
        void Init()
        {
            memset(_outputBuffer, 0, sizeof(BitBlock) * PADDED_OUT_BLOCKS);
            memset(_outputBatchBuffer, 0, sizeof(BitBlock) * BATCH_SIZE * PADDED_OUT_BLOCKS);
        }

        void Save(std::ofstream &fs) {} // 終端
        void Load(std::ifstream &fs) {} // 終端

        const BitBlock *Forward(const BitBlock *netInput)
        {
            // バッファに入力を詰める
            for (int i_out = 0; i_out < COMPRESS_OUT_BLOCKS; i_out++)
            {
                _outputBuffer[i_out] = netInput[i_out];
            }
            return _outputBuffer;
        }

        void ResetWeight()
        {
        }

#pragma region Train
        BitBlock *TrainForward(const BitBlock *netInput)
        {
            // バッファに入力を詰める
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                int batchShiftOut = b * PADDED_OUT_BLOCKS;
                for (int i_out = 0; i_out < COMPRESS_OUT_BLOCKS; i_out++)
                {
                    _outputBatchBuffer[batchShiftOut + i_out] = netInput[batchShiftOut + i_out];
                }
            }
            return _outputBatchBuffer;
        }

        void TrainBackward(const GradientType *nextGrad)
        {
            // 学習する要素無し
        }
#pragma endregion
    };
}

#endif
