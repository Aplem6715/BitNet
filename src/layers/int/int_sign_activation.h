#ifndef INT_SIGN_ACTIVATION_H_INCLUDED_
#define INT_SIGN_ACTIVATION_H_INCLUDED_

#include "../../net_common.h"
#include "../../util/random_util.h"
#include <algorithm>

namespace bitnet
{
	/**
 * @brief Int型符号アクティベーション層
 * 
 * @tparam PreviousLayer_t 前のレイヤー型
 */
	template <typename PreviousLayer_t>
	class IntSignActivation
	{
	public:
		// 出力次元数
		static constexpr int COMPRESS_OUT_DIM = PreviousLayer_t::COMPRESS_OUT_DIM;
		// 入力次元数
		static constexpr int COMPRESS_IN_DIM = COMPRESS_OUT_DIM;

	private:
		// 前の層
		PreviousLayer_t _prevLayer;
		// 前の層に伝播する勾配
		GradientType _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM];
		// 出力バッファ（次の層が参照する
		IntBitType _outputBuffer[COMPRESS_OUT_DIM];
		IntBitType _outputBatchBuffer[BATCH_SIZE * COMPRESS_OUT_DIM];
		int32_t *_inputBatchBuffer;

	public:
		const IntBitType *Forward(const int8_t *netInput)
		{
			const int32_t* input = _prevLayer.Forward(netInput);
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				int32_t x = input[i_out];
				bool isPositive = x > 0;

				_outputBuffer[i_out] = isPositive ? 1 : -1;
			}
			return _outputBuffer;
		}

		void ResetWeight()
		{
			_prevLayer.ResetWeight();
		}

#pragma region Train

		// int32_t -> int_01
		IntBitType *TrainForward(const int8_t *netInput)
		{
			_inputBatchBuffer = _prevLayer.TrainForward(netInput);

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					int32_t x = _inputBatchBuffer[batchShiftOut + i_out];
					const int32_t htanh = std::max(static_cast<int32_t>(-1), std::min(static_cast<int32_t>(1), x));
					double probPositive = (htanh + 1.0) / 2.0;
					bool isPositive = Random::GetReal01() < probPositive;

					_outputBatchBuffer[batchShiftOut + i_out] = isPositive ? 1 : -1;
				}
			}
			return _outputBatchBuffer;
		}

		void TrainBackward(const GradientType *nextGrad)
		{
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i = 0; i < COMPRESS_OUT_DIM; i++)
				{
					GradientType g = nextGrad[batchShiftOut + i];

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
}

#endif
