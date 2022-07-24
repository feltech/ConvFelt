#pragma once
#include <Felt/Impl/Common.hpp>
#include <Felt/Impl/Mixin/GridMixin.hpp>
#include <Felt/Impl/Mixin/NumericMixin.hpp>
#include <Felt/Impl/Mixin/PartitionedMixin.hpp>

#include "Numeric.hpp"

/**
 * This header contains the grid structures used by ConvFelt.
 *
 * Each convolution has a iw*ih*id sized input volume. Each filter then has a cw*ch*id window on
 * that volume, defined by the offset (dw, dh), for a total number of windows ww*wh. So we have a 5D
 * input grid of size ww*wh*(ww*cw)*(wh*ch)*id.
 *
 * A convolution then has an associated output data grid of size (Cw*ww)*(Ch*wh)*Cd, where each
 * filter produces an output grid of size Cw*Ch*Cd resulting from a (cw*ch*id)x(Cw*Ch*Cd) matrix
 * applied to the input window.
 *
 * Each filter has its input data window copied into its own spatial partition. This gives a spatial
 * partition size of (ww, wh, cw, ch, id), that is, it is partitioned only along the input image
 * width and height in steps of (cw, ch), with a (dw, dh) initial offset.
 *
 * The output spatial partition size is the filter's output size (Cw, Ch, Cd).
 */

namespace convfelt
{
template <typename T, Felt::Dim D>
using InputGridTD = Felt::Impl::Grid::Simple<T, D>;
using InputGrid = InputGridTD<convfelt::Scalar, 3>;


template <typename T, Felt::Dim D>
class FilterTD
	: FELT_MIXINS(
		  (FilterTD<T, D>),
		  (Grid::Access::ByValue)(Grid::Activate)(Grid::Data)(Grid::Resize)(Numeric::Snapshot),
		  (Grid::Index))
//{
private:
	using This = FilterTD<T, D>;

	using Traits = Felt::Impl::Traits<This>;
	using Leaf = typename Traits::Leaf;

	using AccessImpl = Felt::Impl::Mixin::Grid::Access::ByValue<This>;
	using ActivateImpl = Felt::Impl::Mixin::Grid::Activate<This>;
	using DataImpl = Felt::Impl::Mixin::Grid::Data<This>;
	using SizeImpl = Felt::Impl::Mixin::Grid::Resize<This>;
	using SnapshotImpl = Felt::Impl::Mixin::Numeric::Snapshot<This>;

public:
	using VecDi = Felt::VecDi<D>;
	using typename SnapshotImpl::ArrayColMap;

	explicit FilterTD(const Leaf background_) : ActivateImpl{background_} {}

	using AccessImpl::get;
	using AccessImpl::index;
	using AccessImpl::set;
	using ActivateImpl::activate;
	using DataImpl::data;
	using SizeImpl::inside;
	using SizeImpl::offset;
	using SizeImpl::resize;
	using SizeImpl::size;
	using SnapshotImpl::array;
	using SnapshotImpl::matrix;
};

using Filter = FilterTD<convfelt::Scalar, 3>;

template <typename T, Felt::Dim D>
class ConvGridTD : FELT_MIXINS(
					 (ConvGridTD<T, D>),
					 (Grid::Size)(Partitioned::Access)(Partitioned::Children)(Partitioned::Leafs),
					 (Grid::Index))
private:
	using This = ConvGridTD<T, D>;

	using Traits = Felt::Impl::Traits<This>;
	using Leaf = typename Traits::Leaf;

	using AccessImpl = Felt::Impl::Mixin::Partitioned::Access<This>;
	using ChildrenImpl = Felt::Impl::Mixin::Partitioned::Children<This>;
	using LeafsImpl = Felt::Impl::Mixin::Partitioned::Leafs<This>;
	using SizeImpl = Felt::Impl::Mixin::Grid::Size<This>;

public:
	using ChildrenGrid = typename ChildrenImpl::ChildrenGrid;
	using Child = typename Traits::Child;
	using VecDi = Felt::VecDi<D>;

	ConvGridTD(
		const VecDi & size_,
		const VecDi & offset_,
		const VecDi & child_size_,
		const Leaf background_)
		: SizeImpl{size_, offset_}, ChildrenImpl{size_, offset_, child_size_, Child{background_}}
	{
		for (auto & child : ChildrenImpl::children().data()) child.activate();
	}

	using AccessImpl::get;
	using AccessImpl::set;
	using ChildrenImpl::children;
	using LeafsImpl::leafs;
	using LeafsImpl::pos_child;
	using LeafsImpl::pos_idx_child;
	using SizeImpl::inside;
	using SizeImpl::offset;
	using SizeImpl::size;
};

using ConvGrid = ConvGridTD<convfelt::Scalar, 3>;

}  // namespace ConvFelt

template <typename T, Felt::Dim D>
struct Felt::Impl::Traits<convfelt::ConvGridTD<T, D>>
{
	/// Single index stored in each grid node.
	using Leaf = T;
	/// Dimension of grid.
	static constexpr Dim t_dims = D;

	using Child = convfelt::FilterTD<T, D>;
	using Children = Felt::Impl::Tracked::SingleListSingleIdxByRef<Child, D>;
};

template <typename T, Felt::Dim D>
struct Felt::Impl::Traits<convfelt::FilterTD<T, D>>
{
	using Leaf = T;
	static constexpr Dim t_dims = D;
};