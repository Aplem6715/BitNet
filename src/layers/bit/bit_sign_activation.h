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
#include <algorithm>

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

private:
	// 前の層
	PreviousLayer_t _prevLayer;
	// 前の層に伝播する勾配
	double _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM];
	// 出力バッファ（次の層が参照する
	BitBlock _outputBuffer[PADDED_OUT_BLOCKS];
	BitBlock _outputBatchBuffer[BATCH_SIZE * PADDED_OUT_BLOCKS];
	double *_inputBatchBuffer; // TODO: int化？

public:
	const BitBlock *Forward(const int8_t *netInput)
	{
		// TODO
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
			const int batchShiftOut = b * PADDED_OUT_BLOCKS;
			for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
			{
				// TODO ビット並列化
				const double x = _inputBatchBuffer[batchShiftOut + i_in];
				const double htanh = std::max(-1.0, std::min(1.0, x));
				const double probPositive = (htanh + 1.0) / 2.0;
				const BitBlock isPositive = Random::GetReal01() < probPositive;

				const int blockIdx = GetBlockIndex(i_in);
				const BitBlock bitShift = GetBitIndexInBlock(i_in);
				const BitBlock block = _outputBatchBuffer[batchShiftOut + blockIdx];
				const BitBlock mask = ~(1 << bitShift);
				const BitBlock newBit = isPositive << bitShift;

				_outputBatchBuffer[batchShiftOut + blockIdx] = (block & mask) | newBit;
			}
		}
		return _outputBatchBuffer;
	}

	void TrainBackward(const GradientType *nextGrad)
	{
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			const int batchShiftOut = b * COMPRESS_OUT_DIM;
			for (int i = 0; i < COMPRESS_OUT_DIM; i++)
			{
				const double g = nextGrad[batchShiftOut + i];

				// d_Hard-tanh
				if (std::abs(_inputBatchBuffer[batchShiftOut + i]) <= 1)
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

#endif