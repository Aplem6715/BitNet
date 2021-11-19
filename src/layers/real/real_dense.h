
#ifndef REAL_DENSE_H_INCLUDED_
#define REAL_DENSE_H_INCLUDED_

template <typename PreviousLayer_t, int OutputBits>
class RealDenseLayer
{
public:
	static constexpr int kCompOutDim = OutputBits;
	static constexpr int kCompInDim = PreviousLayer_t::kCompactOutputDim;

private:
public:
};

#endif
