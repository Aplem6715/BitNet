#ifndef INT_DENSE_H_INCLUDED_
#define INT_DENSE_H_INCLUDED_

#include "../../net_common.h"

/**
 * @brief Int型全結合層。-1/1のウェイトを持つ
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
	int _outputBatchBuffer[BATCH_SIZE][COMPRESS_OUT_DIM];
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
		// TODO
	}

#pragma region Train
	const int *TrainForward(const int8_t *netInput)
	{
		// TODO
	}

	void TrainBackward(const GradientType *nextGrad){
		// TODO
	}
#pragma endregion
};

#endif