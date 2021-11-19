#ifndef BIT_DENSE_H_INCLUDED_
#define BIT_DENSE_H_INCLUDED_

template <typename PreviousLayer_t, int OutputBits>
class BitDenseLayer
{
public:
	static constexpr int COMPRESS_OUT_DIM = OutputBits;
	static constexpr int COMPRESS_IN_DIM = PreviousLayer_t::kCompactOutputDim;

private:
public:
};

#endif