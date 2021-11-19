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

	// 出力次元（ニューロン）の数
	static constexpr int COMPRESS_OUT_DIM = OutputBits;

private:
	// 前の層
	PreviousLayer_t _prevLayer;
	// 出力バッファ（次の層が参照する
	double _outputBuffer[COMPRESS_OUT_DIM];
	// 2値重み(-1 or 1)
	BitWeight _weight[COMPRESS_OUT_DIM][PADDED_IN_BLOCKS];
	// バイアス
	BiasType _bias[COMPRESS_OUT_DIM];

#pragma region Train
	// 前の層に伝播する勾配
	double _gradsToPrev[BATCH_SIZE * COMPRESS_IN_DIM];
	// 勾配法用の実数値重み
	double _realWeight[COMPRESS_OUT_DIM][COMPRESS_IN_DIM] = {0};
	// バッチ学習版出力バッファ（学習時はこちらのバッファを使用する
	double _outputBatchBuffer[BATCH_SIZE * COMPRESS_OUT_DIM];
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
		// TODO
		_prevLayer.ResetWeight();
	}

#pragma region Train
	double *TrainForward(const BitBlock* netInput){
		// TODO
		return _outputBatchBuffer;
	}
	
	void TrainBackward(const GradientType *nextGrad){
		// TODO
	}

#pragma endregion
};

#endif