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
#include "felt2/components/eigen.hpp"
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
	using Stream = sycl::stream *;
#else
	using Stream = std::ostream *;
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
	Stream m_stream{&std::cerr};
#endif
};
}  // namespace Grid
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

public:
	using VecDi = VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;

private:
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
		requires(is_device_shared)
		: m_size_impl{size_, offset_},
		  m_data_impl{{std::move(context), std::move(device)}},
		  m_activate_impl{m_size_impl, m_data_impl, m_stream_impl, background_}
	{
		m_activate_impl.activate();
	}

	ByRef(const VecDi & size_, const VecDi & offset_, Leaf background_)
		requires(!is_device_shared)
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
	decltype(auto) get(VecDi const & pos) noexcept
	{
		return m_access_impl.get(pos);
	}
	decltype(auto) get(VecDi const & pos) const noexcept
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

template <typename T, Felt::Dim D>
class FilterTD
{
private:
	using This = FilterTD<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

public:
	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;

	using SizeImpl = felt2::components::ResizableSize<Traits>;
	using StreamImpl = felt2::components::Stream;
	using DataImpl = felt2::components::DataArraySpan<Traits>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, StreamImpl, SizeImpl, DataImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, DataImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, SizeImpl, DataImpl, StreamImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, DataImpl>;

	using DataArray = typename DataImpl::Array;
	using ArrayColMap = typename MatrixImpl::ArrayColMap;

private:
	DataImpl m_data_impl{};
	SizeImpl m_size_impl;
	StreamImpl m_stream_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl};
	AccessImpl m_access_impl{m_size_impl, m_data_impl, m_assert_bounds_impl};
	MatrixImpl const m_matrix_impl{m_data_impl};

public:
	explicit FilterTD(SizeImpl size_impl) : m_size_impl{std::move(size_impl)} {}

	FilterTD(This const & other)
		: m_data_impl{other.m_data_impl},
		  m_size_impl{other.m_size_impl},
		  m_stream_impl{other.m_stream_impl},
		  m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl},
		  m_access_impl{m_size_impl, m_data_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_data_impl}
	{
	}

	FilterTD(This && other) noexcept
		: m_data_impl{std::move(other.m_data_impl)},
		  m_size_impl{std::move(other.m_size_impl)},
		  m_stream_impl{std::move(other.m_stream_impl)},
		  m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl},
		  m_access_impl{m_size_impl, m_data_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_data_impl}
	{
	}

	FilterTD & operator=(FilterTD other) noexcept
	{
		std::swap(*this, other);
		return *this;
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
	decltype(auto) get(VecDi const & pos) noexcept
	{
		return m_access_impl.get(pos);
	}
	decltype(auto) get(VecDi const & pos) const noexcept
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
	decltype(auto) set(auto &&... args) noexcept
	{
		return m_access_impl.set(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) offset(auto &&... args) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) size(auto &&... args) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) resize(auto &&... args) noexcept
	{
		return m_size_impl.resize(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) inside(auto &&... args) const noexcept
	{
		return m_size_impl.inside(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) array(auto &&... args) const noexcept
	{
		return m_matrix_impl.array(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) matrix(auto &&... args) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) set_stream(auto &&... args) noexcept
	{
		return m_stream_impl.set_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) assert_pos_bounds(auto &&... args) const noexcept
	{
		return m_assert_bounds_impl.assert_pos_bounds(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) assert_pos_idx_bounds(auto &&... args) const noexcept
	{
		return m_assert_bounds_impl.assert_pos_idx_bounds(std::forward<decltype(args)>(args)...);
	}
};

using Filter = FilterTD<convfelt::Scalar, 3>;

template <typename T, Felt::Dim D, bool is_device_shared = false>
class ConvGridTD
{
private:
	using This = ConvGridTD<T, D, is_device_shared>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

public:
	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;
	using Child = convfelt::FilterTD<Leaf, Traits::k_dims>;
	using ChildrenGrid = Felt::Impl::ByRef<Child, Traits::k_dims, is_device_shared>;

private:
	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;
	using StreamImpl = felt2::components::Stream;
	using DataImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMDataArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, StreamImpl, SizeImpl, DataImpl>;
	using MatrixImpl = felt2::components::EigenColMajor2DMap<Traits, DataImpl, ChildrenSizeImpl>;

	DataImpl m_data_impl;
	SizeImpl const m_size_impl;
	ChildrenSizeImpl const m_children_size_impl;
	StreamImpl m_stream_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl};
	MatrixImpl m_matrix_impl{m_data_impl, m_children_size_impl};

	ChildrenGrid m_children;

public:
	struct scoped_stream_t
	{
#ifdef SYCL_DEVICE_ONLY
		scoped_stream_t(This & parent, StreamImpl::StreamType * stream)
			: m_parent{parent}, m_prev_stream{&m_parent.get_stream()}
		{
			m_parent.set_stream(stream);
			m_parent.children().set_stream(stream);
			for (auto & child : convfelt::iter::val(m_parent.children())) child.set_stream(stream);
		}

		~scoped_stream_t()
		{
			m_parent.set_stream(m_prev_stream);
			m_parent.children().set_stream(m_prev_stream);
			for (auto & child : convfelt::iter::val(m_parent.children()))
				child.set_stream(m_prev_stream);
		}

	private:
		This & m_parent;
		StreamImpl::StreamType * m_prev_stream;
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

	ConvGridTD(const VecDi & size_, const Felt::VecDi<D - 1> & child_window_)
		: ConvGridTD{size_, calc_child_size(child_window_, size_), {0, 0, 0}}
	{
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_)
		: ConvGridTD{size_, child_size_, {0, 0, 0}}
	{
	}

	ConvGridTD(
		const VecDi & size_,
		const Felt::VecDi<D - 1> & child_window_,
		sycl::context const & context,
		sycl::device const & device)
		requires(is_device_shared)
		: m_data_impl{{context, device}},
		  m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, calc_child_size(child_window_, m_size_impl.size())},
		  m_children{
			  m_children_size_impl.children_size(),
			  VecDi::Zero(),
			  Child{{VecDi::Zero(), VecDi::Zero()}},
			  context,
			  device}
	{
		initialise();
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_, const VecDi & offset_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{
			  m_children_size_impl.children_size(),
			  VecDi::Zero(),
			  Child{{VecDi::Zero(), VecDi::Zero()}}}
	{
		initialise();
	}

	const ChildrenGrid & children() const noexcept
	{
		return m_children;
	}

	ChildrenGrid & children() noexcept
	{
		return m_children;
	}

	decltype(auto) data(auto &&... args) noexcept
	{
		return m_data_impl.data(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) data(auto &&... args) const noexcept
	{
		return m_data_impl.data(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) offset(auto &&... args) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) size(auto &&... args) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) child_size(auto &&... args) const noexcept
	{
		return m_children_size_impl.child_size(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) inside(auto &&... args) const noexcept
	{
		return m_size_impl.inside(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) matrix(auto &&... args) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) matrix(auto &&... args) noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get_stream(auto &&... args) const noexcept
	{
		return m_stream_impl.get_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get_stream(auto &&... args) noexcept
	{
		return m_stream_impl.get_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) set_stream(auto &&... args) noexcept
	{
		return m_stream_impl.set_stream(std::forward<decltype(args)>(args)...);
	}

private:
	void initialise()
	{
		assert(
			m_children_size_impl.child_size()(m_children_size_impl.child_size().size() - 1) ==
				size()(size().size() - 1) &&
			"Depth of children must be same as depth of parent");

		Felt::Vec2u const matrix_size{
			m_children_size_impl.num_elems_per_child(), m_children_size_impl.num_children()};

		m_data_impl.data().resize(matrix_size.prod());
		std::span const all_data{m_data_impl.data()};

		felt2::PosIdx const num_child_idxs = m_children_size_impl.num_elems_per_child();

		for (auto const & [idx, child] : convfelt::iter::idx_and_val(m_children))
		{
			child.data() = all_data.subspan(idx * num_child_idxs, num_child_idxs);
		}

		m_children_size_impl.resize_children(m_children);
	}

	VecDi calc_child_size(const Felt::VecDi<D - 1> & window_, const VecDi & size_)
	{
		VecDi child_size_;
		child_size_ << window_, size_(size_.size() - 1);
		return child_size_;
	}
};

using ConvGrid = ConvGridTD<convfelt::Scalar, 3>;
}  // namespace convfelt

namespace Felt::Impl
{
template <typename T, Felt::Dim D, bool is_device_shared>
struct Traits<convfelt::ConvGridTD<T, D, is_device_shared>>
{
	/// Single index stored in each grid node.
	using Leaf = T;
	/// Dimension of grid.
	static constexpr Dim t_dims = D;
};

template <typename T, Felt::Dim D>
struct Traits<convfelt::FilterTD<T, D>>
{
	using Leaf = T;
	static constexpr Dim t_dims = D;
};

template <typename T, Felt::Dim D, bool is_device_shared>
struct Traits<Felt::Impl::ByRef<T, D, is_device_shared>>
{
	using Leaf = T;
	static constexpr Dim t_dims = D;
};
}  // namespace Felt::Impl
