﻿/**
 * @file int_dense.h
 * @author Daichi Sato
 * @brief Int型全結合層の定義
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

#ifndef INT_DENSE_H_INCLUDED_
#define INT_DENSE_H_INCLUDED_

#include "../../util/random_util.h"
#include "../../net_common.h"

/**
 * @brief Int型全結合層。-1/1のウェイトを持つ。
 * 
 * @tparam PreviousLayer_t 前のレイヤー型
 * @tparam OutputBits 次元数（ニューロン数）
 */
template <typename PreviousLayer_t, int OutputBits>
class IntDenseLayer
{
public:
	// 出力次元（ニューロン）の数
	static constexpr int COMPRESS_OUT_DIM = OutputBits;
	// 入力次元（前の層のニューロン）の数
	static constexpr int COMPRESS_IN_DIM = PreviousLayer_t::kCompactOutputDim;

private:
	// 前の層
	PreviousLayer_t _prev_layer;
	// 出力バッファ（次の層が参照する
	int _outputBuffer[COMPRESS_OUT_DIM];
	// 2値重み(-1 or 1)
	IntBitWeight _weight[COMPRESS_OUT_DIM][COMPRESS_IN_DIM];
	// バイアス
	BiasType _bias[COMPRESS_OUT_DIM];

#pragma region Train
	// 前の層に伝播する勾配
	double _gradsToPrev[COMPRESS_IN_DIM];
	// 勾配法用の実数値重み
	double _realWeight[COMPRESS_OUT_DIM][COMPRESS_IN_DIM];
	// バッチ学習版出力バッファ（学習時はこちらのバッファを使用する
	int _outputBatchBuffer[BATCH_SIZE * COMPRESS_OUT_DIM];
	// 勾配計算用の入力バッファ（実態は前の層の出力バッファを参照するポインタ
	IntBitType *_inputBatchBuffer;
#pragma endregion
public:
	const int *Forward(const int8_t *netInput)
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
		for (int i_out = 0; i_out < COMPRESS_IN_DIM; i_out++)
		{
			_bias[i_out] = 0;
			for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
			{
				double rand = Random::GetReal01() * 2 - 1;
				_realWeight[i_out][i_in] = rand;
				_weight[i_out][i_in] = rand > 0 ? 1 : -1;
			}
		}
		_prev_layer.ResetWeight();
	}

#pragma region Train
	const int *TrainForward(const int8_t *netInput)
	{
		_inputBatchBuffer = _prev_layer.TrainForward(netInput);

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			int batchShiftIn = b * COMPRESS_IN_DIM;
			int batchShiftOut = b * COMPRESS_OUT_DIM;
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				// パディング分も含めて±1積和演算
				int sum = _bias[i_out];
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					sum += _inputBatchBuffer[batchShiftIn + i_in] * _weight[i_out][i_in];
				}

				_outputBatchBuffer[batchShiftOut + i_out] = sum;
			}
		}

		return _outputBatchBuffer;
	}

	void TrainBackward(const GradientType *nextGrad){
		// 勾配更新
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			int batchShiftIn = b * COMPRESS_IN_DIM;
			int batchShiftOut = b * COMPRESS_OUT_DIM;
			for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
			{
				double sum = 0;
				for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
				{
					sum += nextGrad[batchShiftOut + i_out] * _weight[i_out][i_in];
				}
				_gradsToPrev[batchShiftIn + i_in] = sum;
			}
		}

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			int batchShiftIn = b * COMPRESS_IN_DIM;
			int batchShiftOut = b * COMPRESS_OUT_DIM;
			// 重み調整
			for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
			{
				_bias[i_out] += nextGrad[batchShiftOut + i_out];
				for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
				{
					_realWeight[i_out][i_in] += nextGrad[batchShiftOut + i_out] * _inputBatchBuffer[batchShiftIn + i_in];
				}
			}
		}

		// 2値化
		for (int i_out = 0; i_out < COMPRESS_OUT_DIM; i_out++)
		{
			for (int i_in = 0; i_in < COMPRESS_IN_DIM; i_in++)
			{
				// Clipping
				double tmp_w = std::max(-1.0, std::min(1.0, _realWeight[i_out][i_in]));
				_realWeight[i_out][i_in] = tmp_w;

				_weight[i_out][i_in] = (tmp_w > 0) ? 1 : -1;
			}
		}
	}
#pragma endregion
};

#endif