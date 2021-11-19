#ifndef BIT_DENSE_H_INCLUDED_
#define BIT_DENSE_H_INCLUDED_

template <typename PreviousLayer_t, int OutputBits>
class BitDenseLayer
{
public:
	static constexpr int kCompOutDim = OutputBits;
	static constexpr int kCompInDim = PreviousLayer_t::kCompactOutputDim;

private:
public:
};

#endif