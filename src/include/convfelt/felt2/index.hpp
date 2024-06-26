#pragma once
#include "./typedefs.hpp"

namespace felt2
{

/**
 * Get index in data array of position vector.
 *
 * The grid is packed in a 1D array, so this method is required to get the index in that array of
 * the D-dimensional position.
 *
 * @param pos_ position in grid.
 * @param size_ size of grid.
 * @param offset_ spatial offset of grid.
 * @return index in data array of pos in grid of given size and offset.
 */
template <Dim D>
PosIdx index(VecDi<D> const & pos_, VecDi<D> const & size_, VecDi<D> const & offset_)
{
	using AxisCoord = typename VecDi<D>::Scalar;

	AxisCoord pos_idx = 0;

	for (Dim i = 0; i < D; i++)
	{
		AxisCoord pos_idx_axis = pos_(i) - offset_(i);

		for (Dim j = i + 1; j < D; j++) pos_idx_axis *= size_(j);

		pos_idx += pos_idx_axis;
	}
	return PosIdx(pos_idx);
}

/**
 * Get index in data array of position vector.
 *
 * This overload makes use of the properties of size that guaranteed to be a power-of-two. I.e.
 * uses bit shifting operations.
 *
 * The grid is packed in a 1D array, so this method is required to get the index in that array of
 * the D-dimensional position.
 *
 * @param pos_ position in grid.
 * @param size_ size of grid.
 * @param offset_ spatial offset of grid.
 * @return index in data array of pos in grid of given size and offset.
 */
template <Dim D>
PosIdx index(VecDi<D> const & pos_, PowTwoDu<D> const & size_, VecDi<D> const & offset_)
{
	using AxisCoord = typename VecDi<D>::Scalar;

	AxisCoord pos_idx = 0;

	auto const & exps = size_.exps();

	for (Dim i = 0; i < D; ++i)
	{
		AxisCoord idx_i = (pos_(i) - offset_(i));

		for (Dim j = i + 1; j < D; ++j) idx_i <<= exps(j);

		pos_idx += idx_i;
	}
	return static_cast<PosIdx>(pos_idx);
}

/**
 * Get position of index.
 *
 * Given an index and the size and offset of a grid, calculate the position vector that the index
 * pertains to in a representative 1D array.
 *
 * @param idx_ index to query.
 * @param size_ size of grid.
 * @param offset_ spatial offset of grid.
 * @return position that the given index would represent in a grid of given size and offset.
 */
template <Dim D>
VecDi<D> index(PosIdx idx_, VecDi<D> const & size_, VecDi<D> const & offset_ = VecDi<D>::Zero())
{
	/*
	Eg. 2D: row major order (3x4=12): (x,y)[idx] =>
	(0,0)[0], (0,1)[1], (0,2)[2],  (0,3)[3]
	(1,0)[4], (1,1)[5], (1,2)[6],  (1,3)[7]
	(2,0)[8], (2,1)[9], (2,2)[10], (2,3)[11]

	E.g. 3D:
	z = idx % Dz
	y = (idx/Dz) % Dy
	x = (idx/Dz)/Dy % Dx
	*/
	using AxisCoord = typename VecDi<D>::Scalar;
	VecDi<D> pos;

	// Note: since `Dim` is unsigned, we cannot allow `axis` to decrement below zero.
	for (Dim axis = D - 1; axis != 0; axis--)
	{
		pos(axis) = AxisCoord(idx_) % size_(axis) + offset_(axis);
		idx_ /= PosIdx(size_(axis));
	}
	pos(0) = AxisCoord(idx_) % size_(0) + offset_(0);

	return pos;
}


/**
 * Get position of index.
 *
 * Given an index and the size and offset of a grid, calculate the position vector that the index
 * pertains to in a representative 1D array.
 *
 * This overload makes use of the properties of size that guaranteed to be a power-of-two. I.e.
 * uses bit shifting operations.
 *
 * @param idx_ index to query.
 * @param size_ size of grid.
 * @param offset_ spatial offset of grid.
 * @return position that the given index would represent in a grid of given size and offset.
 */
template <Dim D>
VecDi<D> index(PosIdx idx_, PowTwoDu<D> const & size_, VecDi<D> const & offset_ = VecDi<D>::Zero())
{
	VecDi<D> pos;
	auto const & mask = size_.mask();
	auto const & exps = size_.exps();

	// Note: since `Dim` is unsigned, we cannot allow `axis` to decrement below zero.
	for (Dim axis = D - 1; axis != 0; --axis)
	{
		pos(axis) = static_cast<felt2::AxisPos>(idx_ & mask(axis)) + offset_(axis);
		idx_ >>= exps(axis);  // i.e. /= size(axis)
	}
	pos(0) = static_cast<felt2::AxisPos>(idx_ & mask(0)) + offset_(0);

	return pos;
}

/**
 * Test if a position is inside given bounds.
 *
 * @tparam TPos the type of position vector.
 * @tparam TBounds the type of boundary (min/max) vectors.
 * @param pos_ position in grid to query.
 * @param pos_min_ minimum allowed position.
 * @param pos_max_ one more than the maximum allowed position.
 * @return true if position lies inside the grid, false otherwise.
 */
template <typename TPos, typename TBounds>
bool inside(TPos const & pos_, TBounds const & pos_min_, TBounds const & pos_max_)
{
	for (Dim i = 0; i < pos_.size(); i++)
	{
		if (static_cast<Scalar>(pos_(i)) >= static_cast<Scalar>(pos_max_(i)))
			return false;
		if (static_cast<Scalar>(pos_(i)) < static_cast<Scalar>(pos_min_(i)))
			return false;
	}
	return true;
}
}  // namespace felt2