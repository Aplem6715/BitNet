#ifndef RANDOM_UTIL_H_INCLUDED_
#define RANDOM_UTIL_H_INCLUDED_

#include <random>

namespace Random
{
	extern std::random_device rnd;
	extern std::mt19937 mt;
	extern std::uniform_real_distribution<double> rnd_prob01;

	inline void Seed(int seed)
	{
		Random::mt = std::mt19937(seed);
	}

	// [0~1)の実数乱数を取得
	inline double GetReal01()
	{
		return Random::rnd_prob01(Random::mt);
	}

	inline uint32_t GetUInt()
	{
		return Random::mt();
	}
}

#endif