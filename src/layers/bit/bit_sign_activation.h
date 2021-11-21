/**
 * @file bit_sign_activation.h
 * @author Daichi Sato
 * @brief 符号関数アクティベーション層。
 * @version 1.0
 * @date 2021-11-20
 * 
 * @copyright Copyright (c) 2021 Daichi Sato
 * 
 * 符号関数アクティベーション
 * 必要であれば出力に対してビット演算用のパディングを追加する
 * 
 */
#ifndef BIT_SIGN_ACTIVATION_H_
#define BIT_SIGN_ACTIVATION_H_

#include "../../net_common.h"
#include "../../util/random_util.h"
#include "../../util/bit_helper.h"
#include <algorithm>

namespace bitnet
{
	/**
	 * @brief Int型符号アクティベーション層
	 * 
	 * @tparam PreviousLayer_t 前のレイヤー型
	 */
	template <typename PreviousLayer_t>
	class BitSignActivation
	{
	public:
		// 出力次元数
		static constexpr int COMPRESS_OUT_DIM = PreviousLayer_t::COMPRESS_OUT_DIM;
		static constexpr int COMPRESS_OUT_BITS = COMPRESS_OUT_DIM;
		static constexpr int COMPRESS_OUT_BLOCKS = BitToBlockCount(COMPRESS_OUT_DIM);
		static constexpr int PADDED_OUT_BITS = AddPaddingToBitSize(COMPRESS_OUT_BITS);
		static constexpr int PADDED_OUT_BLOCKS = BitToBlockCount(PADDED_OUT_BITS);
		// 入力次元数
		static constexpr int COMPRESS_IN_DIM = COMPRESS_OUT_DIM;
		static constexpr int PADDED_IN_BLOCKS = AddPaddingToBytes(COMPRESS_IN_DIM);

	private:
		// 前の層
		PreviousLayer_t _prevLayer;
		// 前の層に伝播する勾配
		double _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM] = {0};
		// 出力バッファ（次の層が参照する
		alignas(__m256i) BitBlock _outputBuffer[PADDED_OUT_BLOCKS] = {0};
		alignas(__m256i) BitBlock _outputBatchBuffer[BATCH_SIZE * PADDED_OUT_BLOCKS] = {0};
		int8_t *_inputBuffer;
		int8_t *_inputBatchBuffer;

	public:
		const BitBlock *Forward(const BitBlock *netInput)
		{
			_inputBuffer = _prevLayer.Forward(netInput);

			// TODO 検証
			CollectSignBit(_inputBuffer, (int *)_outputBuffer, PADDED_IN_BLOCKS);

			return _outputBuffer;
		}

		void ResetWeight()
		{
			_prevLayer.ResetWeight();
		}

#pragma region Train

		// double -> int_01
		BitBlock *TrainForward(const BitBlock *netInput)
		{
			_inputBatchBuffer = _prevLayer.TrainForward(netInput);

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				const int batchShiftIn = b * PADDED_IN_BLOCKS;
				const int batchShiftOut = b * PADDED_OUT_BLOCKS;
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					// TODO ビット並列化
					const int8_t x = _inputBatchBuffer[batchShiftIn + i_in];
					const int8_t htanh = std::max(static_cast<int8_t>(-1), std::min(static_cast<int8_t>(1), x));
					const double probPositive = (htanh + 1) / 2.0;
					const double rand = Random::GetReal01();
					const BitBlock isPositive = (rand < probPositive) ? 1 : 0;

					const int blockIdx = GetBlockIndex(i_in);
					const int bitShift = GetBitIndexInBlock(i_in);
					const BitBlock block = _outputBatchBuffer[batchShiftOut + blockIdx];
					const BitBlock mask = ~(1 << bitShift);
					const BitBlock newBit = isPositive << bitShift;
					const BitBlock result = (block & mask) | newBit;

					_outputBatchBuffer[batchShiftOut + blockIdx] = result;
				}
			}
			return _outputBatchBuffer;
		}

		void TrainBackward(const GradientType *nextGrad)
		{
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				const int batchShiftIn = b * PADDED_IN_BLOCKS;
				const int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i = 0; i < COMPRESS_OUT_DIM; i++)
				{
					const double g = nextGrad[batchShiftOut + i];

					// d_Hard-tanh
					if (std::abs(_inputBatchBuffer[batchShiftIn + i]) <= 1)
					{
						_gradsToPrev[batchShiftOut + i] = g;
					}
					else
					{
						_gradsToPrev[batchShiftOut + i] = 0;
					}
				}
			}
			_prevLayer.TrainBackward(_gradsToPrev);
		}

#pragma endregion
	};
}

#endif
