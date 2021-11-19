
#ifndef REAL_DENSE_H_INCLUDED_
#define REAL_DENSE_H_INCLUDED_

template <typename PreviousLayer_t, int OutputBits>
class RealDenseLayer
{
public:
	static constexpr int COMPRESS_OUT_DIM = OutputBits;
	static constexpr int COMPRESS_IN_DIM = PreviousLayer_t::kCompactOutputDim;

private:
public:
};

#endif
