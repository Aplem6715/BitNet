#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "util/make_data.h"

namespace int_net
{
	using Input = IntInputLayer<8>;
	using Hidden1 = IntSignActivation<IntDenseLayer<Input, 128>>;
	using Hidden2 = IntSignActivation<IntDenseLayer<Hidden1, 32>>;
	using Output = IntDenseLayer<Hidden2, 1>;
	using Network = Output;
}

void TrainInt(int_net::Network &net, int nbTrain)
{
	int8_t inputData[BATCH_SIZE * 8];
	int8_t teacherData[BATCH_SIZE];
	double diffs[BATCH_SIZE];
	double lr = 0.001;
	double maeSum = 0;
	double scale = 32;
	for (int train = 0; train < nbTrain; train++)
	{
		util::MakePopBatch(BATCH_SIZE, scale, inputData, teacherData);
		// util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);

		const double *pred = net.TrainForward(inputData);

		double mae;
		double mse = util::CalcSquaredError(BATCH_SIZE, 1, scale, lr, pred, teacherData, diffs, &mae);

		net.TrainBackward(diffs);
		// std::cout << mae << std::endl;
		maeSum += mae;
	}
	std::cout << maeSum / scale / nbTrain << std::endl;
}

int main()
{
	int_net::Network net;
	net.ResetWeight();

	Random::Seed(42);
	for (int i = 0; i < 100; i++)
	{
		TrainInt(net, 2000);
	}

	int8_t inputData[BATCH_SIZE * 8];
	int8_t teacherData[BATCH_SIZE];
	double scale = 32;
	for (int i = 0; i < 10; i++)
	{
		util::MakePopBatch(BATCH_SIZE, scale, inputData, teacherData);
		const double *pred = net.TrainForward(inputData);
		std::cout << (double)teacherData[0] << ":" << pred[0] << std::endl;
	}
	return 0;
}