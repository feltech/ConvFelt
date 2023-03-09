#pragma once
#include <concepts>
#include <span>

#include <sycl/sycl.hpp>

#include <Felt/Impl/Common.hpp>
#include <Felt/Impl/Mixin/GridMixin.hpp>
#include <Felt/Impl/Mixin/NumericMixin.hpp>
#include <Felt/Impl/Mixin/PartitionedMixin.hpp>
#include <Felt/Impl/Util.hpp>

#include "assert_compat.hpp"
#include "felt2/components/core.hpp"
#include "felt2/components/sycl.hpp"
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
};

template <class TDerived>
class AssertBounds
{
	using Traits = Impl::Traits<TDerived>;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	static constexpr Dim t_dims = Traits::t_dims;
	using VecDi = Felt::VecDi<t_dims>;

protected:
#ifdef SYCL_DEVICE_ONLY
	using Stream = sycl::stream*;
#else
	using Stream = std::ostream*;
#endif

	[[nodiscard]] Stream get_stream() const
	{
		return m_stream;
	}

	void set_stream(Stream stream)
	{
		m_stream = stream;
	}

	void assert_pos_bounds(const Felt::PosIdx pos_idx_, const char * title_) const
	{
		assert_pos_idx_bounds(pos_idx_, title_);
	}

	void assert_pos_idx_bounds(const VecDi & pos_, const char * title_) const
	{
		assert_pos_bounds(pos_, title_);
	}

	void assert_pos_bounds(const VecDi & pos_, const char * title_) const
	{
		if (m_stream && !pself->inside(pos_))
		{
			*m_stream << "AssertionError: " << title_ << " assert_pos_bounds(" << pos_(0);
			for (Felt::TupleIdx axis = 1; axis < pos_.size(); ++axis)
				*m_stream << ", " << pos_(axis);
			*m_stream << ")\n";
		}
		assert(pself->inside(pos_));
	}

	void assert_pos_idx_bounds(const PosIdx pos_idx_, const char * title_) const
	{
		if (m_stream && pos_idx_ >= pself->data().size())
		{
			auto pos = pself->index(pos_idx_);
			*m_stream << "AssertionError: " << title_ << " assert_pos_idx_bounds(" << pos_idx_
					  << ") i.e. (" << pos(0);
			for (Felt::TupleIdx axis = 1; axis < pos.size(); ++axis) *m_stream << ", " << pos(axis);
			*m_stream << ")\n";
		}
		assert(pos_idx_ < pself->data().size());
	}

private:
#ifdef SYCL_DEVICE_ONLY
	Stream m_stream{nullptr};
#else
	Stream  m_stream{&std::cerr};
#endif
};
}	// namespace Grid
}  // namespace Mixin

template <typename T, Dim D, bool is_device_shared = false>
class ByRef
{
private:
	using This = ByRef<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr Dim k_dims = D;
	};

	using VecDi = VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StreamImpl = felt2::components::Stream;
	using DataImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMDataArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, StreamImpl, SizeImpl, DataImpl>;
	using AccessImpl = felt2::components::AccessByRef<Traits, SizeImpl, DataImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, SizeImpl, DataImpl, StreamImpl>;

	SizeImpl const m_size_impl;
	StreamImpl m_stream_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl};
	AccessImpl m_access_impl{m_size_impl, m_data_impl, m_assert_bounds_impl};
	DataImpl m_data_impl;
	ActivateImpl m_activate_impl;

public:
	ByRef(
		const VecDi & size_,
		const VecDi & offset_,
		Leaf background_,
		sycl::context context,
		sycl::device device)
		requires (is_device_shared)
		: m_size_impl{size_, offset_},
		  m_data_impl{{std::move(context), std::move(device)}},
		  m_activate_impl{m_size_impl, m_data_impl, m_stream_impl, background_}
	{
		m_activate_impl.activate();
	}

	ByRef(const VecDi & size_, const VecDi & offset_, Leaf background_)
		requires (!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_data_impl{},
		  m_activate_impl{m_size_impl, m_data_impl, m_stream_impl, background_}
	{
		m_activate_impl.activate();
	}
	decltype(auto) data(auto &&... args) noexcept
	{
		return m_data_impl.data(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) data(auto &&... args) const noexcept
	{
		return m_data_impl.data(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) index(auto &&... args) const noexcept
	{
		return m_size_impl.index(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get(VecDi const& pos) noexcept
	{
		return m_access_impl.get(pos);
	}
	decltype(auto) get(VecDi const& pos) const noexcept
	{
		return m_access_impl.get(pos);
	}
	decltype(auto) get(auto &&... args) noexcept
	{
		return m_access_impl.get(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get(auto &&... args) const noexcept
	{
		return m_access_impl.get(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) offset(auto &&... args) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) size(auto &&... args) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) set_stream(auto &&... args) noexcept
	{
		return m_stream_impl.set_stream(std::forward<decltype(args)>(args)...);
	}
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
			  Partitioned::Leafs)(Numeric::PartitionedAsColMajorMatrix)(Grid::AssertBounds),
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
	using IndexImpl = Felt::Impl::Mixin::Grid::Index<This>;
	using AssertBoundsImpl = Felt::Impl::Mixin::Grid::AssertBounds<This>;

public:
	using ChildrenGrid = typename ChildrenImpl::ChildrenGrid;
	using Child = typename Traits::Child;
	using VecDi = Felt::VecDi<D>;

	struct scoped_stream_t
	{
#ifdef SYCL_DEVICE_ONLY
		scoped_stream_t(
			This & parent, AssertBoundsImpl::Stream stream)
			: m_parent{parent}, m_prev_stream{m_parent.get_stream()}
		{
			m_parent.set_stream(stream);
			m_parent.children().set_stream(stream);
			for (auto& child : convfelt::iter::val(m_parent.children()))
				child.set_stream(stream);
		}

		~scoped_stream_t()
		{
			m_parent.set_stream(m_prev_stream);
			m_parent.children().set_stream(m_prev_stream);
			for (auto& child : convfelt::iter::val(m_parent.children()))
				child.set_stream(m_prev_stream);
		}

	private:
		This & m_parent;
		AssertBoundsImpl::Stream m_prev_stream;
#endif
	};

	scoped_stream_t scoped_stream([[maybe_unused]] sycl::stream * stream)
	{
#ifdef SYCL_DEVICE_ONLY
		return {*this, stream};
#else
		return {};
#endif
	};

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
	using AssertBoundsImpl::assert_pos_bounds;
	using AssertBoundsImpl::assert_pos_idx_bounds;
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
class FilterTD : FELT_MIXINS(
					 (FilterTD<T, D>),
					 (Grid::Access::ByValue)(Grid::Activate)(Grid::DataSpan)(Grid::Resize)(
						 Numeric::Snapshot)(Grid::AssertBounds),
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
	using IndexImpl = Felt::Impl::Mixin::Grid::Index<This>;
	using AssertBoundsImpl = Felt::Impl::Mixin::Grid::AssertBounds<This>;

public:
	using typename DataSpanImpl::DataArray;
	using typename SnapshotImpl::ArrayColMap;
	using VecDi = Felt::VecDi<D>;

	using AccessImpl::get;
	using AccessImpl::index;
	using AccessImpl::set;
	using AssertBoundsImpl::set_stream;
	using AssertBoundsImpl::assert_pos_bounds;
	using AssertBoundsImpl::assert_pos_idx_bounds;
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
	using Children = Felt::Impl::ByRef<Child, D, !std::is_same_v<A<T>, std::allocator<T>>>;
};

template <typename T, Felt::Dim D>
struct Traits<convfelt::FilterTD<T, D>>
{
	using Leaf = T;
	using DataArray = std::span<Leaf>;
	static constexpr Dim t_dims = D;
};

template <typename T, Felt::Dim D, bool is_device_shared>
struct Traits<Felt::Impl::ByRef<T, D, is_device_shared>>
{
	using Leaf = T;
	static constexpr Dim t_dims = D;
};
}  // namespace Felt::Impl
