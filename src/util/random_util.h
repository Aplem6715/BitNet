#ifndef RANDOM_UTIL_H_INCLUDED_
#define RANDOM_UTIL_H_INCLUDED_

#include <random>

namespace Random
{
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<double> rnd_prob01(0.0, 1.0);

	void Seed(int seed)
	{
		Random::mt = std::mt19937(seed);
	}

	// [0~1)の実数乱数を取得
	double GetReal01()
	{
		return Random::rnd_prob01(Random::mt);
	}

	uint32_t GetUInt()
	{
		return Random::mt();
	}
}

#endif