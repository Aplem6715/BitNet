#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "util/make_data.h"

namespace int_net
{
	using Input = IntInputLayer<2>;
	using Hidden1 = IntSignActivation<IntDenseLayer<Input, 64>>;
	using Hidden2 = IntSignActivation<IntDenseLayer<Hidden1, 32>>;
	using Output = IntDenseLayer<Hidden2, 1>;
	using Network = Output;
}

void TrainInt(int nbTrain)
{
	double lr = 0.0001;
	int_net::Network net;
	net.ResetWeight();
	for (int train = 0; train < nbTrain; train++)
	{
		int8_t inputData[BATCH_SIZE * 2];
		int8_t teacherData[BATCH_SIZE];
		double diffs[BATCH_SIZE];

		util::MakeXORBatch(BATCH_SIZE, 32, inputData, teacherData);

		const double *pred = net.TrainForward(inputData);

		double mae;
		double mse = util::CalcSquaredError(BATCH_SIZE, 1, 32, lr, pred, teacherData, diffs, &mae);

		net.TrainBackward(diffs);
		std::cout << mae << std::endl;
	}
}

int main(){

	TrainInt(100);

	return 0;
}