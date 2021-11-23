﻿/**
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

#include <type_traits>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include "../../util/random_util.h"
#include "../../net_common.h"
#include "../../util/bit_helper.h"

namespace bitnet
{
	/**
	 * @brief ビット演算全結合層。bitBlock入力double出力
	 * 
	 * @tparam PreviousLayer_t 前の層の型
	 * @tparam OutputBits 出力次元数（ニューロン数
	 * @tparam isOutputLayer 出力層ならtrue(default:false)
	 */
	template <typename PreviousLayer_t, int OutputBits, bool isOutputLayer = false>
	class BitDenseLayer
	{
	public:
		using OutputType = typename std::conditional<isOutputLayer, int32_t, int8_t>::type;
		// 入力次元（前の層のニューロン）の数
		static constexpr int COMPRESS_IN_DIM = PreviousLayer_t::COMPRESS_OUT_DIM;
		static constexpr int PADDED_IN_BITS = AddPaddingToBitSize(COMPRESS_IN_DIM);
		static constexpr int PADDED_IN_BLOCKS = BitToBlockCount(PADDED_IN_BITS);
		static constexpr int PADDING_BITS = PADDED_IN_BITS - COMPRESS_IN_DIM;

		// 出力次元（ニューロン）の数
		static constexpr int COMPRESS_OUT_DIM = OutputBits;
		static constexpr int PADDED_OUT_BLOCKS = AddPaddingToBytes(COMPRESS_OUT_DIM);

	private:
#pragma region Train
		// 勾配法用の実数値重み
		alignas(32) float _realWeight[COMPRESS_OUT_DIM][COMPRESS_IN_DIM] = {0};
		// バッチ学習版出力バッファ（学習時はこちらのバッファを使用する
		alignas(32) OutputType _outputBatchBuffer[BATCH_SIZE * PADDED_OUT_BLOCKS] = {0};
#pragma endregion
		// 出力バッファ（次の層が参照する
		alignas(32) OutputType _outputBuffer[PADDED_OUT_BLOCKS] = {0};
		// 2値重み(-1 or 1)
		alignas(32) BitWeight _weight[COMPRESS_OUT_DIM][PADDED_IN_BLOCKS] = {0};
		// 前の層
		PreviousLayer_t _prevLayer;
		// バイアス
		int _bias[COMPRESS_OUT_DIM] = {0};

#pragma region Train
		// 前の層に伝播する勾配
		GradientType _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM] = {0};
		// 勾配法用の実数値バイアス
		double _realBias[COMPRESS_OUT_DIM] = {0};
		// TODO 整数化
		// 勾配計算用の入力バッファ（実態は前の層の出力バッファを参照するポインタ
		BitBlock *_inputBatchBuffer;
#pragma endregion

	public:
		void Init()
		{
			memset(_outputBatchBuffer, 0, sizeof(OutputType) * BATCH_SIZE * PADDED_OUT_BLOCKS);
			memset(_outputBuffer, 0, sizeof(OutputType) * PADDED_OUT_BLOCKS);
			memset(_weight, 0, sizeof(BitWeight) * COMPRESS_OUT_DIM * PADDED_IN_BLOCKS);
			_prevLayer.Init();
		}

		void Save(std::ofstream &fs)
		{
			int dim = COMPRESS_OUT_DIM;
			fs.write(reinterpret_cast<char *>(&dim), sizeof(int));
			fs.write(reinterpret_cast<char *>(_realBias), sizeof(double) * COMPRESS_OUT_DIM);
			fs.write(reinterpret_cast<char *>(_realWeight), sizeof(float) * COMPRESS_OUT_DIM * COMPRESS_IN_DIM);

			_prevLayer.Save(fs);
		}

		void Load(std::ifstream &fs)
		{
			int dim;
			fs.read(reinterpret_cast<char *>(&dim), sizeof(int));

			if (dim != COMPRESS_OUT_DIM)
			{
				throw std::runtime_error("Invalid Model   code dim:" + std::to_string(COMPRESS_OUT_DIM) + "load dim:" + std::to_string(dim));
			}

			fs.read(reinterpret_cast<char *>(_realBias), sizeof(double) * COMPRESS_OUT_DIM);
			fs.read(reinterpret_cast<char *>(_realWeight), sizeof(float) * COMPRESS_OUT_DIM * COMPRESS_IN_DIM);

			_prevLayer.Load(fs);
		}

		const OutputType *Forward(const BitBlock *netInput)
		{
			const BitBlock *input = _prevLayer.Forward(netInput);

			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				int32_t pop;
				// パディング分も含めて±1積和演算
				if (USE_AVX_MADD)
				{
					pop = MaddPopcnt2(input, _weight[i_out], PADDED_IN_BITS);
				}
				else
				{
					pop = 0;
					for (int block = 0; block < PADDED_IN_BLOCKS; block++)
					{
						const BitBlock xnor = ~(input[block] ^ _weight[i_out][block]);
						pop += __popcnt64(xnor);
					}
				}

				// 1,-1の合計値に（[plus] - (bitWidth - [minus]) => 2x[1の数] - bitWidth)
				const int32_t sum = 2 * (pop - PADDING_BITS) - COMPRESS_IN_DIM;
				const int32_t result = sum + _bias[i_out];
				if (isOutputLayer)
				{
					// 出力層ではパディングの必要がない
					_outputBuffer[i_out] = static_cast<OutputType>(result);
				}
				else
				{
					// 次のsign層で符号ビットが分かればいい（32bitのMSBが8bitMSBに来るようにシフト）
					_outputBuffer[i_out] = static_cast<OutputType>(result > 0);
				}
			}

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
				_realBias[i_out] = 0;
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
		OutputType *TrainForward(const BitBlock *netInput)
		{
			_inputBatchBuffer = _prevLayer.TrainForward(netInput);

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftInBlock = b * PADDED_IN_BLOCKS;
				int batchShiftOut = b * PADDED_OUT_BLOCKS;
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					int32_t pop;
					// パディング分も含めて±1積和演算
					if (USE_AVX_MADD)
					{
						pop = MaddPopcnt2(&_inputBatchBuffer[batchShiftInBlock], _weight[i_out], PADDED_IN_BITS);
					}
					else
					{
						pop = 0;
						for (int block = 0; block < PADDED_IN_BLOCKS; block++)
						{
							const BitBlock xnor = ~(_inputBatchBuffer[batchShiftInBlock + block] ^ _weight[i_out][block]);
							pop += __popcnt64(xnor);
						}
					}

					// 1,-1の合計値に（[plus] - (bitWidth - [minus]) => 2x[1の数] - bitWidth)
					const int32_t sum = 2 * (pop - PADDING_BITS) - COMPRESS_IN_DIM;
					const int32_t result = sum + _bias[i_out];
					if (isOutputLayer)
					{
						// 出力層ではパディングの必要がない
						_outputBatchBuffer[b * COMPRESS_OUT_DIM + i_out] = static_cast<OutputType>(result);
					}
					else
					{
						// 算術シフトのコンパイラでのみ正常動作する
						static_assert((((int32_t)0xffffffff >> 1) == 0xffffffff));
						static_assert((((int32_t)0x00000001 >> 1) == 0x00000000));
						constexpr int32_t MSB32 = 1 << 31;
						// 次のsign層で符号ビットが分かればいい（32bitのMSBが8bitMSBに来るようにシフト）
						_outputBatchBuffer[batchShiftOut + i_out] = static_cast<OutputType>((result & MSB32) >> 24 | result /*1と0の区別をつけるため，推論時は不要？*/);
					}
				}
			}

			return _outputBatchBuffer;
		}

		void TrainBackward(const GradientType *nextGrad)
		{
			// 勾配更新
			UpdateGrad(nextGrad);

			UpdateWeights(nextGrad);

			// 2値化
			Binarize();

			_prevLayer.TrainBackward(_gradsToPrev);
		}

		void UpdateGrad(const GradientType *nextGrad)
		{
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int batchShiftIn = b * COMPRESS_IN_DIM;
				int batchShiftOut = b * COMPRESS_OUT_DIM;
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					GradientType sum = 0;
					for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
					{
						sum += nextGrad[batchShiftOut + i_out] * sgn(_realWeight[i_out][i_in]);
					}
					_gradsToPrev[batchShiftIn + i_in] = sum;
				}
			}
		}

		void UpdateWeights(const GradientType *nextGrad)
		{
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				const int batchShiftInBlock = b * PADDED_IN_BLOCKS;
				const int batchShiftOut = b * COMPRESS_OUT_DIM;
				// 重み調整
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					float grad = nextGrad[batchShiftOut + i_out];
					if (grad == 0)
					{
						continue;
					}

					_realBias[i_out] += grad;
					// NegateAddFloats(_realWeight[i_out], grad, &_inputBatchBuffer[batchShiftInBlock], COMPRESS_IN_DIM);
					float *const realWeight = _realWeight[i_out];
					for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
					{
						int blockIdx = GetBlockIndex(i_in);
						int bitShift = GetBitIndexInBlock(i_in);
						if ((_inputBatchBuffer[batchShiftInBlock + blockIdx] >> bitShift) & 1)
						{
							realWeight[i_in] += grad;
						}
						else
						{
							realWeight[i_in] -= grad;
						}
					}
				}
			}
		}

		void Binarize()
		{
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				_bias[i_out] = _realBias[i_out];
				if (COMPRESS_IN_DIM % BYTE_BIT_WIDTH == 0)
				{
					// float-8個分のMSBを読んで
					int cursor = 0;
					for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in += BYTE_BIT_WIDTH)
					{
						float8 packed = _mm256_load_ps(&(_realWeight[i_out][i_in]));
						_weight[i_out][cursor] = ~_mm256_movemask_ps(packed);
						++cursor;
					}
				}
				else
				{
					for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
					{
						// Clipping
						const double tmp_w = std::max(-1.0, std::min(1.0, (double)_realWeight[i_out][i_in]));
						_realWeight[i_out][i_in] = tmp_w;

						const int blockIdx = GetBlockIndex(i_in);
						const int bitShift = GetBitIndexInBlock(i_in);
						const BitBlock block = _weight[i_out][blockIdx];
						const BitBlock mask = ~(1 << bitShift);
						const BitBlock newBit = ((uint8_t)(tmp_w > 0)) << bitShift;
						_weight[i_out][blockIdx] = (block & mask) | newBit;
					}
				}
			}
		}

#pragma endregion
	};
}
#endif
