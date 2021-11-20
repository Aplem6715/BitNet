/**
 * @file bit_dense.h
 * @author Daichi Sato
 * @brief ビット演算全結合層の定義
 * @version 0.1
 * @date 2021-11-19
 * 
 * @copyright Copyright (c) 2021 Daichi Sato
 * 
 * すべての層は次の層内にメンバ変数として展開される。
 * テンプレートによって次元数などを決定するため，メモリ確保は行わず高速動作する。
 * また，テンプレートによる展開を行うため，ヘッダー内にすべての関数を定義している。
 * 
 */

#ifndef BIT_DENSE_H_INCLUDED_
#define BIT_DENSE_H_INCLUDED_

#include "../../util/random_util.h"
#include "../../net_common.h"
#include "../../util/bit_helper.h"
#include <cmath>

namespace bitnet
{
	/**
	 * @brief ビット演算全結合層。bitBlock入力double出力
	 * 
	 * @tparam PreviousLayer_t 前の層の型
	 * @tparam OutputBits 出力次元数（ニューロン数
	 */
	template <typename PreviousLayer_t, int OutputBits>
	class BitDenseLayer
	{
	public:
		// 入力次元（前の層のニューロン）の数
		static constexpr int COMPRESS_IN_DIM = PreviousLayer_t::COMPRESS_OUT_DIM;
		static constexpr int PADDED_IN_BITS = AddPaddingToBitSize(COMPRESS_IN_DIM);
		static constexpr int PADDED_IN_BLOCKS = BitToBlockCount(PADDED_IN_BITS);
		static constexpr int PADDING_BITS = PADDED_IN_BITS - COMPRESS_IN_DIM;

		// 出力次元（ニューロン）の数
		static constexpr int COMPRESS_OUT_DIM = OutputBits;
		// 入力側と表記を統一するための変数。COMPRESS_OUT_DIMと値は同じ
		static constexpr int PADDED_OUT_BLOCKS = COMPRESS_OUT_DIM;

	private:
		// 前の層
		PreviousLayer_t _prevLayer;
		// 出力バッファ（次の層が参照する
		double _outputBuffer[COMPRESS_OUT_DIM] = {0};
		// 2値重み(-1 or 1)
		BitWeight _weight[COMPRESS_OUT_DIM][PADDED_IN_BLOCKS] = {0};
		// バイアス
		BiasType _bias[COMPRESS_OUT_DIM] = {0};

#pragma region Train
		// 前の層に伝播する勾配
		double _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM] = {0};
		// 勾配法用の実数値重み
		double _realWeight[COMPRESS_OUT_DIM][COMPRESS_IN_DIM] = {0};
		// バッチ学習版出力バッファ（学習時はこちらのバッファを使用する
		double _outputBatchBuffer[BATCH_SIZE * COMPRESS_OUT_DIM] = {0};
		// 勾配計算用の入力バッファ（実態は前の層の出力バッファを参照するポインタ
		BitBlock *_inputBatchBuffer;
#pragma endregion

	public:
		const double *Forward(const BitBlock *netInput)
		{
			// TODO
			return _outputBuffer;
		}

		void ClearWeight()
		{
			// TODO
		}

		void ResetWeight()
		{
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				_bias[i_out] = 0;
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					int blockIdx = GetBlockIndex(i_in);
					int bitShift = GetBitIndexInBlock(i_in);
					// Clipping
					double tmp_w = Random::GetReal01() * 2 - 1;
					_realWeight[i_out][i_in] = tmp_w;

					BitBlock block = _weight[i_out][blockIdx];
					BitBlock mask = ~(1 << bitShift);
					BitBlock newBit = (BitBlock)(tmp_w > 0 ? 1 : 0) << bitShift;
					_weight[i_out][blockIdx] = (block & mask) | newBit;
				}
			}

			_prevLayer.ResetWeight();
		}

#pragma region Train
		double *TrainForward(const BitBlock *netInput)
		{
			_inputBatchBuffer = _prevLayer.TrainForward(netInput);

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftInBlock = b * PADDED_IN_BLOCKS;
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					// パディング分も含めて±1積和演算
					int pop = 0;
					for (int block = 0; block < PADDED_IN_BLOCKS; block++)
					{
						const BitBlock xnor = ~(_inputBatchBuffer[batchShiftInBlock + block] ^ _weight[i_out][block]);
						pop += __popcnt64(xnor);
					}

					// 1,-1の合計値に（[plus] - (bitWidth - [minus]) => 2x[1の数] - bitWidth)
					int sum = 2 * (pop - PADDING_BITS) - COMPRESS_IN_DIM;
					_outputBatchBuffer[batchShiftOut + i_out] = (sum + (int)_bias[i_out]);
				}
			}

			return _outputBatchBuffer;
		}

		void TrainBackward(const GradientType *nextGrad)
		{
			// 勾配更新
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftIn = b * COMPRESS_IN_DIM;
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					const int blockIdx = GetBlockIndex(i_in);
					const int bitShift = GetBitIndexInBlock(i_in);
					double sum = 0;
					for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
					{
						const BitBlock bits = _weight[i_out][blockIdx];
						const BitBlock w_bit = (bits >> bitShift) & 0b1;
						const int weight = (w_bit == 1 ? 1 : -1);
						sum += nextGrad[batchShiftOut + i_out] * weight;
					}
					_gradsToPrev[batchShiftIn + i_in] = sum;
				}
			}

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				const int batchShiftInBlock = b * PADDED_IN_BLOCKS;
				const int batchShiftOut = b * COMPRESS_OUT_DIM;
				// 重み調整
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					_bias[i_out] += nextGrad[batchShiftOut + i_out];
					for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
					{
						const int blockIdx = GetBlockIndex(i_in);
						const int bitShift = GetBitIndexInBlock(i_in);
						const BitBlock input = (BitBlock)(_inputBatchBuffer[batchShiftInBlock + blockIdx] >> bitShift) & 1;
						const int8_t sign = (input == 0 ? -1 : 1);
						_realWeight[i_out][i_in] += nextGrad[batchShiftOut + i_out] * sign;
					}
				}
			}

			// 2値化
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					// Clipping
					const double tmp_w = std::max(-1.0, std::min(1.0, _realWeight[i_out][i_in]));
					_realWeight[i_out][i_in] = tmp_w;

					const int blockIdx = GetBlockIndex(i_in);
					const int bitShift = GetBitIndexInBlock(i_in);
					const BitBlock block = _weight[i_out][blockIdx];
					const BitBlock mask = ~(1 << bitShift);
					const BitBlock newBit = ((uint8_t)(tmp_w > 0)) << bitShift;
					_weight[i_out][blockIdx] = (block & mask) | newBit;
				}
			}

			_prevLayer.TrainBackward(_gradsToPrev);
		}

#pragma endregion
	};
}
#endif
