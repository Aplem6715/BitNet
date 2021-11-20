#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "train.h"

int main()
{
	// int_net::Network net;
	bit_net::Network net;
	net.ResetWeight();

	Random::Seed(42);
	for (int i = 0; i < 100; i++)
	{
		// Train<int_net::Network>(net, 100, false);
		train::Train<bit_net::Network>(net, 100, 16, true);
	}
	double diffs[10];
	train::Test<bit_net::Network>(net, 10, 16, diffs);

	return 0;
}