﻿#ifndef MAKE_DATA_H_INCLUDED_
#define MAKE_DATA_H_INCLUDED_

#include "random_util.h"
#include "../net_common.h"
#include "bit_helper.h"
#include <cstdint>

namespace bitnet
{
	namespace util
	{
		void BinarizeInputData(int batchSize, int numData, const int8_t *inputData, BitBlock *binDataOut)
		{
			const int padded_blocks = BitToBlockCount(AddPaddingToBitSize(numData));

			// 0クリア
			for (int b = 0; b < batchSize; b++)
			{
				const int batchShift = b * padded_blocks;
				for (int block = 0; block < padded_blocks; block++)
				{
					binDataOut[batchShift + block] = 0;
				}
			}

			for (int b = 0; b < batchSize; b++)
			{
				const int batchShiftIn = b * numData;
				const int batchShiftBlock = b * padded_blocks;
				for (int i = 0; i < numData; i++)
				{
					int blockIdx = GetBlockIndex(i);
					int bitShift = GetBitIndexInBlock(i);
					uint8_t bit = inputData[batchShiftIn + i] > 0 ? 1 : 0;

					binDataOut[batchShiftBlock + blockIdx] |= bit << bitShift;
				}
			}
		}

		void MakeXORBatch(int batchSize, double tScale, int8_t *inputData, int8_t *teacherData)
		{
			constexpr int INPUT_SIZE = 2;
			for (int b = 0; b < batchSize; b++)
			{
				int batchShift = b * INPUT_SIZE;
				int8_t x1 = Random::GetUInt() % 2;
				int8_t x2 = Random::GetUInt() % 2;
				int8_t t = x1 ^ x2;

				inputData[batchShift + 0] = (x1 == 0) ? -1 : 1;
				inputData[batchShift + 1] = (x2 == 0) ? -1 : 1;

				t = (t == 0) ? -1 : 1;
				teacherData[b] = t * tScale;
			}
		}

		void MakePopBatch(int batchSize, double tScale, int8_t *inputData, int8_t *teacherData)
		{
			constexpr int INPUT_SIZE = 8;
			for (int b = 0; b < batchSize; b++)
			{
				int batchShift = b * INPUT_SIZE;
				int8_t t = 0;
				for (int i = 0; i < INPUT_SIZE; i++)
				{
					inputData[batchShift + i] = ((Random::GetUInt() % 2) == 1) ? 1 : -1;
					if (inputData[batchShift + i] > 0)
					{
						t ^= 1;
					}
				}

				// teacherData[b] = t / (double)INPUT_SIZE * tScale*2 - tScale;
				teacherData[b] = t * (double)tScale - tScale / 2;
			}
		}

		// TODO: ファイル移動
		/**
	 * @brief 勾配と2乗平均誤差を計算する
	 * 
	 * @param batchSize 
	 * @param predSize 
	 * @param predData 
	 * @param teacherData 
	 * @param diffOuts 
	 * @param mae 絶対値平均誤差出力
	 * @return double 
	 */
		double CalcSquaredError(int batchSize, int predSize, double tScale, double lr, const int32_t *predData, const int8_t *teacherData, double *diffOuts, double *maeOut)
		{
			double totalLoss = 0;
			double totalAE = 0;
			for (int b = 0; b < batchSize; b++)
			{
				int batchShift = b * predSize;
				for (int i = 0; i < predSize; i++)
				{
					int idx = batchShift + i;
					double y = predData[idx];
					double t = teacherData[idx];
					double diff = lr * (t - y);
					diffOuts[idx] = diff;

					double ae = std::abs(y - t);
					totalAE += ae;
					totalLoss += ae * ae / tScale;
				}
			}

			*maeOut = totalAE / (batchSize * predSize);
			totalLoss /= 2; // 2乗平均誤差の微分関数美化用の係数分
			return (totalLoss / 2.0) / (batchSize * predSize);
		}
	}
}

#endif
