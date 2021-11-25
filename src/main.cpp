#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "train.h"

int main()
{
	using namespace bitnet;

	// int_net::Network net;
	BitNetwork net;
	net.ResetWeight();

	Random::Seed(42);
	for (int i = 0; i < 100; i++)
	{
		// Train<int_net::Network>(net, 100, false);
		Train<BitNetwork>(net, 100, 16, true);
	}
	float diffs[25];
	Test<BitNetwork>(net, 25, 16, true, false, diffs);

	return 0;
}