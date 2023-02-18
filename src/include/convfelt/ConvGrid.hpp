#pragma once
#include <concepts>
#include <span>

#include <sycl/sycl.hpp>

#include <Felt/Impl/Common.hpp>
#include <Felt/Impl/Mixin/GridMixin.hpp>
#include <Felt/Impl/Mixin/NumericMixin.hpp>
#include <Felt/Impl/Mixin/PartitionedMixin.hpp>
#include <Felt/Impl/Util.hpp>

#include "Numeric.hpp"
#include "iter.hpp"

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
template <class Traits>
concept uses_usm_allocator = std::same_as<
	typename Traits::Allocator,
	sycl::usm_allocator<typename Traits::Leaf, sycl::usm::alloc::shared>>;
}

namespace Felt::Impl
{
namespace Mixin
{
namespace Numeric
{
template <class TDerived>
class PartitionedAsColMajorMatrix
{
private:
	using Traits = Impl::Traits<TDerived>;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	static constexpr Dim t_dims = Traits::t_dims;
	using VecDi = Felt::VecDi<t_dims>;

	Felt::NodeIdx const m_child_size;
	Felt::NodeIdx const m_num_children;

protected:
	using MatrixMap =
		Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0>;
	using MatrixConstMap =
		Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const, 0>;

	PartitionedAsColMajorMatrix(VecDi const & child_size, VecDi const & num_children)
		: m_child_size{child_size.prod()}, m_num_children{num_children.prod()}
	{
	}

	/**
	 * Map the raw data to a (column-major) Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	MatrixMap matrix()
	{
		return MatrixMap{pself->data().data(), m_child_size, m_num_children};
	}

	MatrixConstMap matrix() const
	{
		return MatrixConstMap{pself->data().data(), m_child_size, m_num_children};
	}
};
}  // namespace Numeric

namespace Grid
{

template <class TDerived>
class DataSpan
{
private:
	using Traits = Impl::Traits<TDerived>;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	static constexpr Dim t_dims = Traits::t_dims;
	using VecDi = Felt::VecDi<t_dims>;

protected:
	using DataArray = typename Traits::DataArray;
	DataArray m_data;

protected:
	DataSpan(sycl::context ctx, sycl::device dev)
		requires convfelt::uses_usm_allocator<Traits>
		: m_data{
			  sycl::usm_allocator<Leaf, sycl::usm::alloc::shared>{std::move(ctx), std::move(dev)}}
	{
	}

	DataSpan() = default;

	DataArray & data()
	{
		return m_data;
	}

	const DataArray & data() const
	{
		return m_data;
	}

	/**
	 * Serialisation hook for cereal library.
	 *
	 * @param ar
	 */
	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(m_data);
	}

//	/**
//	 * Check if given position's index is within the data array and raise a domain_error if not.
//	 *
//	 * @param pos_idx_ position in grid to query.
//	 * @param title_ message to include in generated exception.
//	 */
//	void assert_pos_idx_bounds(const PosIdx pos_idx_, std::string_view title_) const
//	{
//		assert_pos_idx_bounds(pself->index(pos_idx_), title_);
//	}
//
//	/**
//	 * Check if given position's index is within the data array and raise a domain_error if not.
//	 *
//	 * @param pos_ position in grid to query.
//	 * @param title_ message to include in generated exception.
//	 */
//	void assert_pos_idx_bounds(const VecDi & pos_, std::string_view title_) const
//	{
//		const PosIdx pos_idx = pself->index(pos_);
//
//		if (pos_idx > pself->data().size())
//		{
//			const VecDi & pos_min = pself->offset();
//			const VecDi & pos_max = (pself->size() + pos_min - VecDi::Constant(1));
//			std::stringstream err;
//			err << title_ << format(pos_.transpose()) << " data index " << pos_idx
//				<< " is greater than data size " << pself->data().size() << " for grid "
//				<< format(pos_min) << "-" << format(pos_max) << std::endl;
//			std::string err_str = err.str();
//			throw std::domain_error(err_str);
//		}
//
//		pself->assert_pos_bounds(pos_, title_);
//	}
};

}  // namespace Grid
}  // namespace Mixin

template <typename T, Dim D, template <typename> class A>
class ByRef : FELT_MIXINS(
				  (ByRef<T, D, A>),
				  (Grid::Access::ByRef)(Grid::Activate)(Grid::DataSpan)(Grid::Size),
				  (Grid::Index))
//{
private:
	using This = ByRef<T, D, A>;
	using Traits = Impl::Traits<This>;

	using AccessImpl = Impl::Mixin::Grid::Access::ByRef<This>;
	using ActivateImpl = Impl::Mixin::Grid::Activate<This>;
	using DataImpl = Impl::Mixin::Grid::DataSpan<This>;
	using SizeImpl = Impl::Mixin::Grid::Size<This>;

	using VecDi = Felt::VecDi<Traits::t_dims>;
	using Leaf = typename Traits::Leaf;

public:
	ByRef(const VecDi & size_, const VecDi & offset_, Leaf background_)
		: ActivateImpl{std::move(background_)}, SizeImpl{size_, offset_}
	{
		ActivateImpl::activate();
	}

	ByRef(
		const VecDi & size_,
		const VecDi & offset_,
		Leaf background_,
		sycl::context context,
		sycl::device device)
		requires convfelt::uses_usm_allocator<Traits>
		: ActivateImpl{std::move(background_)}, DataImpl{context, device}, SizeImpl{size_, offset_}
	{
		ActivateImpl::activate();
	}

	using AccessImpl::get;
	using AccessImpl::index;
	using ActivateImpl::activate;
	using DataImpl::assert_pos_idx_bounds;
	using DataImpl::data;
	using SizeImpl::offset;
	using SizeImpl::size;
};
}  // namespace Felt::Impl

namespace convfelt
{
template <typename T, Felt::Dim D>
using InputGridTD = Felt::Impl::Grid::Simple<T, D>;
using InputGrid = InputGridTD<convfelt::Scalar, 3>;

template <typename T, Felt::Dim D, template <typename> class A = std::allocator>
class ConvGridTD
	: FELT_MIXINS(
		  (ConvGridTD<T, D, A>),
		  (Grid::Activate)(Grid::DataSpan)(Grid::Size)(Partitioned::Access)(Partitioned::Children)(
			  Partitioned::Leafs)(Numeric::PartitionedAsColMajorMatrix),
		  (Grid::Index))
//{
private:
	using This = ConvGridTD<T, D, A>;
	using Traits = Felt::Impl::Traits<This>;

	using AccessImpl = Felt::Impl::Mixin::Partitioned::Access<This>;
	using ChildrenImpl = Felt::Impl::Mixin::Partitioned::Children<This>;
	using DataImpl = Felt::Impl::Mixin::Grid::DataSpan<This>;
	using LeafsImpl = Felt::Impl::Mixin::Partitioned::Leafs<This>;
	using PartitionedAsColMajorMatrixImpl =
		Felt::Impl::Mixin::Numeric::PartitionedAsColMajorMatrix<This>;
	using SizeImpl = Felt::Impl::Mixin::Grid::Size<This>;

public:
	using ChildrenGrid = typename ChildrenImpl::ChildrenGrid;
	using Child = typename Traits::Child;
	using VecDi = Felt::VecDi<D>;

	ConvGridTD(const VecDi & size_, const Felt::VecDi<D - 1> & child_size_)
		: ConvGridTD{size_, calc_child_size(child_size_, size_), {0, 0, 0}}
	{
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_)
		: ConvGridTD{size_, child_size_, {0, 0, 0}}
	{
	}

	ConvGridTD(
		const VecDi & size_,
		const Felt::VecDi<D - 1> & child_size_,
		sycl::context context,
		sycl::device device)
		requires uses_usm_allocator<Traits>
		: DataImpl{context, device},
		  SizeImpl{size_, {0, 0, 0}},
		  ChildrenImpl{
			  size_, {0, 0, 0}, calc_child_size(child_size_, size_), Child{}, context, device},
		  PartitionedAsColMajorMatrixImpl{
			  ChildrenImpl::child_size(), ChildrenImpl::children().size()}
	{
		initialise();
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_, const VecDi & offset_)
		: SizeImpl{size_, offset_},
		  ChildrenImpl{size_, offset_, child_size_, Child{}},
		  PartitionedAsColMajorMatrixImpl{
			  ChildrenImpl::child_size(), ChildrenImpl::children().size()}
	{
		initialise();
	}

	using AccessImpl::get;
	using AccessImpl::set;
	using ChildrenImpl::child_size;
	using ChildrenImpl::children;
	using DataImpl::data;
	using LeafsImpl::leafs;
	using LeafsImpl::pos_child;
	using LeafsImpl::pos_idx_child;
	using PartitionedAsColMajorMatrixImpl::matrix;
	using SizeImpl::inside;
	using SizeImpl::offset;
	using SizeImpl::size;

private:
	void initialise()
	{
		assert(
			ChildrenImpl::child_size()(ChildrenImpl::child_size().size() - 1) ==
			size()(size().size() - 1));

		Felt::Vec2u const padded_matrix_size = calc_padded_matrix_size();

		data().resize(padded_matrix_size.prod());
		std::span const all_data{data()};

		auto const num_child_idxs = static_cast<Felt::PosIdx>(child_size().prod());
		auto const num_padded_child_idxs = static_cast<Felt::PosIdx>(padded_matrix_size(0));

		for (auto const & [idx, child] : convfelt::iter::idx_and_val(children()))
		{
			child.data() = all_data.subspan(idx * num_padded_child_idxs, num_child_idxs);
		}
	}

	VecDi calc_child_size(const Felt::VecDi<D - 1> & window_, const VecDi & size_)
	{
		VecDi child_size_;
		child_size_ << window_, size_(size_.size() - 1);
		return child_size_;
	}

	Felt::Vec2u calc_padded_matrix_size()
	{
		return {child_size().prod(), children().size().prod()};
	}
};

using ConvGrid = ConvGridTD<convfelt::Scalar, 3>;

template <typename T, Felt::Dim D>
class FilterTD
	: FELT_MIXINS(
		  (FilterTD<T, D>),
		  (Grid::Access::ByValue)(Grid::Activate)(Grid::DataSpan)(Grid::Resize)(Numeric::Snapshot),
		  (Grid::Index))
//{
private:
	using This = FilterTD<T, D>;

	using Traits = Felt::Impl::Traits<This>;
	using Leaf = typename Traits::Leaf;

	using AccessImpl = Felt::Impl::Mixin::Grid::Access::ByValue<This>;
	using DataSpanImpl = Felt::Impl::Mixin::Grid::DataSpan<This>;
	using SizeImpl = Felt::Impl::Mixin::Grid::Resize<This>;
	using SnapshotImpl = Felt::Impl::Mixin::Numeric::Snapshot<This>;

public:
	using typename DataSpanImpl::DataArray;
	using typename SnapshotImpl::ArrayColMap;
	using VecDi = Felt::VecDi<D>;

	using AccessImpl::get;
	using AccessImpl::index;
	using AccessImpl::set;
	using DataSpanImpl::data;
	using SizeImpl::inside;
	using SizeImpl::offset;
	using SizeImpl::resize;
	using SizeImpl::size;
	using SnapshotImpl::array;
	using SnapshotImpl::matrix;
};

using Filter = FilterTD<convfelt::Scalar, 3>;

}  // namespace convfelt

namespace Felt::Impl
{
template <typename T, Felt::Dim D, template <typename> class A>
struct Traits<convfelt::ConvGridTD<T, D, A>>
{
	/// Single index stored in each grid node.
	using Leaf = T;
	using Allocator = A<T>;
	using DataArray = std::vector<Leaf, Allocator>;
	/// Dimension of grid.
	static constexpr Dim t_dims = D;

	using Child = convfelt::FilterTD<T, D>;
	using Children = Felt::Impl::ByRef<Child, D, A>;
};

template <typename T, Felt::Dim D>
struct Traits<convfelt::FilterTD<T, D>>
{
	using Leaf = T;
	using DataArray = std::span<Leaf>;
	static constexpr Dim t_dims = D;
};

/**
 * Traits for Tracked::ByRef.
 *
 * @tparam T type of data to store in the grid.
 * @tparam D number of dimensions of the grid.
 */
template <typename T, Felt::Dim D, template <typename> class A>
struct Traits<Felt::Impl::ByRef<T, D, A>>
{
	/// Single index stored in each grid node.
	using Leaf = T;
	using Allocator = A<T>;
	using DataArray = std::vector<Leaf, Allocator>;
	/// Dimension of grid.
	static constexpr Dim t_dims = D;
};
}  // namespace Felt::Impl
