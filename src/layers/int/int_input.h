#ifndef INT_INPUT_H_INCLUDED_
#define INT_INPUT_H_INCLUDED_

#include "../../net_common.h"
namespace bitnet
{

	/**
 * @brief Int型入力層
 * 
 * @tparam PreviousLayer_t 前のレイヤー型
 * @tparam OutputBits 入力数
 */
	template <int InputBits>
	class IntInputLayer
	{
	public:
		// 出力次元数
		static constexpr int COMPRESS_OUT_DIM = InputBits;
		// 入力次元数
		static constexpr int COMPRESS_IN_DIM = InputBits;

	private:
		// 出力バッファ（次の層が参照する
		IntBitType _outputBuffer[COMPRESS_OUT_DIM];
		IntBitType _outputBatchBuffer[BATCH_SIZE * COMPRESS_OUT_DIM];

	public:
		const IntBitType *Forward(const int8_t *netInput)
		{
			// バッファに入力を詰める
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				_outputBuffer[i_out] = netInput[i_out];
			}
			return _outputBuffer;
		}

		void ResetWeight()
		{
		}

#pragma region Train
		IntBitType *TrainForward(const int8_t *netInput)
		{
			// バッファに入力を詰める
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
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