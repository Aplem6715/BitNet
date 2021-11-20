#include <iostream>
#include "layers/layers.h"
#include "net_common.h"
#include "train.h"

int main()
{
	using namespace bitnet;

	// int_net::Network net;
	IntNetwork net;
	net.ResetWeight();

	Random::Seed(42);
	for (int i = 0; i < 100; i++)
	{
		// Train<int_net::Network>(net, 100, false);
		Train<IntNetwork>(net, 100, 16, true);
	}
	double diffs[10];
	Test<IntNetwork>(net, 10, 16, false, false, diffs);

	return 0;
}