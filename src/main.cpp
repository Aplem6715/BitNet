#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "util/make_data.h"

namespace int_net
{
	using Input = IntInputLayer<2>;
	using Hidden1 = IntSignActivation<IntDenseLayer<Input, 32>>;
	using Hidden2 = IntSignActivation<IntDenseLayer<Hidden1, 16>>;
	using Output = IntDenseLayer<Hidden2, 1>;
	using Network = Output;
}

namespace bit_net
{
	using Input = BitInputLayer<2>;
	using Hidden1 = BitSignActivation<BitDenseLayer<Input, 32>>;
	using Hidden2 = BitSignActivation<BitDenseLayer<Hidden1, 16>>;
	using Output = BitDenseLayer<Hidden2, 1>;
	using Network = Output;
}

template <typename NetType>
double *TrainForward(NetType &net, const int8_t *input, const uint8_t *binInput);

template <>
double *TrainForward<int_net::Network>(int_net::Network &net, const int8_t *input, const uint8_t *binInput)
{
	return net.TrainForward(input);
}

template <>
double *TrainForward<bit_net::Network>(bit_net::Network &net, const int8_t *input, const uint8_t *binInput)
{
	return net.TrainForward(binInput);
}

template <typename NetType>
void Train(NetType &net, int nbTrain, bool shouldBitInput)
{
	constexpr int dataSize = 2;
	constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
	int8_t inputData[BATCH_SIZE * dataSize];
	int8_t teacherData[BATCH_SIZE];
	BitBlock binInput[BATCH_SIZE * padded_blocks];
	double diffs[BATCH_SIZE];
	double lr = 0.001;
	double maeSum = 0;
	double scale = 16;
	for (int train = 0; train < nbTrain; train++)
	{
		util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);
		// util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);

		if (shouldBitInput)
		{
			util::BinarizeInputData(BATCH_SIZE, dataSize, inputData, binInput);
		}

		const double *pred = TrainForward<NetType>(net, inputData, binInput);

		double mae;
		double mse = util::CalcSquaredError(BATCH_SIZE, 1, scale, lr, pred, teacherData, diffs, &mae);

		net.TrainBackward(diffs);
		// std::cout << *pred << std::endl;
		maeSum += mae;
	}
	std::cout << maeSum / scale / nbTrain << std::endl;
}

int main()
{
	// int_net::Network net;
	bit_net::Network net;
	net.ResetWeight();

	Random::Seed(42);
	for (int i = 0; i < 100; i++)
	{
		// Train<int_net::Network>(net, 100, false);
		Train<bit_net::Network>(net, 100, true);
	}

	constexpr int dataSize = 2;
	constexpr int padded_blocks = BitToBlockCount(AddPaddingToBitSize(dataSize));
	int8_t inputData[BATCH_SIZE * 2];
	BitBlock binInput[BATCH_SIZE * padded_blocks];
	int8_t teacherData[BATCH_SIZE];
	double scale = 16;
	for (int i = 0; i < 10; i++)
	{
		util::MakeXORBatch(BATCH_SIZE, scale, inputData, teacherData);
		// const double *pred = net.TrainForward(inputData);
		const double *pred = net.TrainForward(binInput);
		std::cout << (double)teacherData[0] << ":" << pred[0] << std::endl;
	}
	return 0;
}