#pragma once
#include <concepts>
#include <span>

#include <sycl/sycl.hpp>

#include "assert_compat.hpp"

#include "felt2/typedefs.hpp"

#include "felt2/components/core.hpp"
#include "felt2/components/eigen.hpp"
#include "felt2/components/sycl.hpp"

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

template <typename T, felt2::Dim D, bool is_device_shared = false>
class ByValue
{
public:
	using This = ByValue<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StreamImpl = felt2::components::Stream;
	using DataImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMDataArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, StreamImpl, SizeImpl, DataImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, DataImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, SizeImpl, DataImpl, StreamImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, DataImpl>;

private:
	SizeImpl const m_size_impl;
	StreamImpl m_stream_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl};
	AccessImpl m_access_impl{m_size_impl, m_data_impl, m_assert_bounds_impl};
	DataImpl m_data_impl;
	ActivateImpl m_activate_impl;
	MatrixImpl const m_matrix_impl{m_data_impl};

public:
	ByValue(
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

	ByValue(const VecDi & size_, const VecDi & offset_, Leaf background_)
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
};

template <typename T, felt2::Dim D>
using InputGridTD = ByValue<T, D>;
using InputGrid = InputGridTD<felt2::Scalar, 3>;

template <typename T, felt2::Dim D, bool is_device_shared = false>
class ByRef
{
public:
	using This = ByRef<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
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

private:
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

template <typename T, felt2::Dim D>
class FilterTD
{
public:
	using This = FilterTD<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;

	using SizeImpl = felt2::components::ResizableSize<Traits>;
	using StreamImpl = felt2::components::Stream;
	using DataImpl = felt2::components::DataArraySpan<Traits>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, StreamImpl, SizeImpl, DataImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, DataImpl, AssertBoundsImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, DataImpl>;

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

using Filter = FilterTD<felt2::Scalar, 3>;

template <typename T, felt2::Dim D, bool is_device_shared = false>
class ConvGridTD
{
public:
	using This = ConvGridTD<T, D, is_device_shared>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = Traits::Leaf;
	using Child = FilterTD<Leaf, Traits::k_dims>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, is_device_shared>;

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

private:
	DataImpl m_data_impl;
	SizeImpl const m_size_impl;
	ChildrenSizeImpl const m_children_size_impl;
	StreamImpl m_stream_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_stream_impl, m_size_impl, m_data_impl};
	MatrixImpl m_matrix_impl{m_data_impl, m_children_size_impl};

	ChildrenGrid m_children;

public:
	ConvGridTD(const VecDi & size_, const felt2::VecDi<D - 1> & child_window_)
		: ConvGridTD{size_, window_to_size(child_window_, size_), {0, 0, 0}}
	{
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_)
		: ConvGridTD{size_, child_size_, {0, 0, 0}}
	{
	}

	ConvGridTD(
		const VecDi & size_,
		const felt2::VecDi<D - 1> & child_window_,
		sycl::context const & context,
		sycl::device const & device)
		requires(is_device_shared)
		: m_data_impl{{context, device}},
		  m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, window_to_size(child_window_, m_size_impl.size())},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_data_impl, Child{{VecDi::Zero(), VecDi::Zero()}}, context, device)}
	{
		assert_child_size();
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_, const VecDi & offset_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_data_impl, Child{{VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
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
	decltype(auto) has_stream(auto &&... args) const noexcept
	{
		return m_stream_impl.has_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get_stream(auto &&... args) const noexcept
	{
		return m_stream_impl.get_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) get_stream(auto &&... args) noexcept
	{
		return m_stream_impl.get_stream(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) set_stream(sycl::stream * stream) noexcept
	{
		m_children.set_stream(stream);
		for (auto & child : convfelt::iter::val(m_children))
			child.set_stream(stream);
		return m_stream_impl.set_stream(stream);
	}

private:
	void assert_child_size()
	{
		assert(
			m_children_size_impl.child_size()(m_children_size_impl.child_size().size() - 1) ==
				m_size_impl.size()(m_size_impl.size().size() - 1) &&
			"Depth of children must be same as depth of parent");
	}

	VecDi window_to_size(const felt2::VecDi<D - 1> & window_, const VecDi & size_)
	{
		VecDi child_size_;
		child_size_ << window_, size_(size_.size() - 1);
		return child_size_;
	}
};

using ConvGrid = ConvGridTD<felt2::Scalar, 3>;
}  // namespace convfelt