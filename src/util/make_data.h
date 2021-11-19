﻿#ifndef MAKE_DATA_H_INCLUDED_
#define MAKE_DATA_H_INCLUDED_

#include "random_util.h"
#include <cstdint>

namespace util
{
	void MakeXORBatch(int batchSize, double tScale, int8_t *inputData, int8_t *teacherData)
	{
		constexpr int INPUT_SIZE = 2;
		for (int b = 0; b < batchSize; b++)
		{
			int batchShift = b * INPUT_SIZE;
			int8_t x1 = Random::GetReal01() > 0.5;
			int8_t x2 = Random::GetReal01() > 0.5;
			int8_t t = x1 ^ x2;

			inputData[batchShift + 0] = (x1 == 0) ? -1 : 1;
			inputData[batchShift + 1] = (x2 == 0) ? -1 : 1;

			t = (t == 0) ? -1 : 1;
			teacherData[b] = t * tScale;
		}
	}

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
	double CalcSquaredError(int batchSize, int predSize, double tScale, double lr, const double *predData, const int8_t *teacherData, double *diffOuts, double *maeOut)
	{
		double totalLoss = 0;
		double totalAE = 0;
		for (int b = 0; b < batchSize; b++)
		{
			int batchShift = b * predSize;
			for (int i = 0; i < predSize; i++)
			{
				double t = teacherData[batchShift + i];
				double y = predData[batchShift + i];
				double diff = lr * (t - y) / tScale;
				diffOuts[batchShift + i] = diff;

				double ae = std::abs(y - t) / tScale;
				totalAE += ae;
				totalLoss += ae * ae;
			}
		}

		*maeOut = totalAE / (batchSize * predSize);
		totalLoss /= 2; // 2乗平均誤差の微分関数美化用の係数分
		return (totalLoss / 2.0) / (batchSize * predSize);
	}
}

#endif