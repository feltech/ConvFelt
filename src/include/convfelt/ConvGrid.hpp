#pragma once
#include <cassert>
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

using GridFlags = std::uint8_t;

struct GridFlag
{
	enum : GridFlags
	{
		is_device_shared = 1 << 0,
		is_child = 1 << 1
	};
};

template <GridFlags flags>
using host_or_device_context_t = std::conditional_t<
	static_cast<bool>(flags & GridFlag::is_device_shared),
	// device grid
	felt2::components::device::
		Context<felt2::components::device::Log, felt2::components::device::Aborter>,
	// host grid
	felt2::components::Context<felt2::components::Log, felt2::components::Aborter>>;

template <class T, GridFlags flags>
using ref_if_child_t =
	std::conditional_t<static_cast<bool>(flags & GridFlag::is_child), std::reference_wrapper<T>, T>;

constexpr decltype(auto) unwrap_ref(auto && maybe_wrapped)
{
	return static_cast<std::add_lvalue_reference_t<
		std::unwrap_reference_t<std::remove_reference_t<decltype(maybe_wrapped)>>>>(maybe_wrapped);
}

auto make_host_context()
{
	return felt2::components::Context{felt2::components::Log{}, felt2::components::Aborter{}};
}

auto make_device_context(sycl::device const & dev, sycl::context const & ctx)
{
	return felt2::components::device::Context{
		felt2::components::device::Log{}, felt2::components::device::Aborter{}, dev, ctx};
}

auto make_device_context(sycl::queue const & queue)
{
	return make_device_context(queue.get_device(), queue.get_context());
}

class USMMatrix
{
	struct Traits
	{
		using Leaf = felt2::Scalar;
	};
	using StorageImpl = felt2::components::device::USMRawArray<Traits>;
	using MatrixImpl = felt2::components::MatrixMap<StorageImpl>;
	using BytesImpl = felt2::components::StorageBytes<StorageImpl>;

public:
	USMMatrix(
		felt2::Dim const rows,
		felt2::Dim const cols,
		sycl::device const & dev,
		sycl::context const & ctx)
		: m_storage_impl{static_cast<std::size_t>(rows * cols), dev, ctx},
		  m_bytes_impl{m_storage_impl},
		  m_matrix_impl{rows, cols, m_storage_impl}
	{
	}

	[[nodiscard]] decltype(auto) matrix(auto &&... args) noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args)>(args)...);
	}

	[[nodiscard]] decltype(auto) matrix(auto &&... args) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args)>(args)...);
	}

	[[nodiscard]] decltype(auto) bytes(auto &&... args) noexcept
	{
		return m_bytes_impl.bytes(std::forward<decltype(args)>(args)...);
	}

	[[nodiscard]] decltype(auto) bytes(auto &&... args) const noexcept
	{
		return m_bytes_impl.bytes(std::forward<decltype(args)>(args)...);
	}

	operator MatrixImpl::Matrix()
	{  // NOLINT(*-explicit-constructor)
		return m_matrix_impl.matrix();
	}

private:
	StorageImpl m_storage_impl;
	BytesImpl m_bytes_impl;
	MatrixImpl const m_matrix_impl;
};

template <typename T, felt2::Dim D, GridFlags flags = 0>
class ByValue
{
public:
	using This = ByValue<T, D, flags>;
	static constexpr bool is_device_shared = flags & GridFlag::is_device_shared;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using ContextImpl = host_or_device_context_t<flags>;
	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	ref_if_child_t<ContextImpl, flags> m_context_impl;
	SizeImpl const m_size_impl;
	StorageImpl m_storage_impl;
	ActivateImpl m_activate_impl;
	AssertBoundsImpl const m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	MatrixImpl const m_matrix_impl{m_storage_impl};

public:
	ByValue(
		decltype(m_context_impl) context_,
		const VecDi & size,
		const VecDi & offset,
		Leaf background) requires(is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size, offset},
		  m_storage_impl{{context().context(), context().device()}},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, std::move(background)}
	{
		m_activate_impl.activate();
	}

	ByValue(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const VecDi & offset_,
		Leaf background_) requires(!is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_storage_impl{},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, std::move(background_)}
	{
		m_activate_impl.activate();
	}

	// Note: if/when copying is required, cannot default due to reference copying.
	ByValue(This const & other) noexcept = delete;

	ByValue(This && other) noexcept = default;

	~ByValue() = default;

	This & operator=(This && other) noexcept = default;
	This & operator=(This const & other) = delete;

	auto & context(this auto & self) noexcept
	{
		return unwrap_ref(self.m_context_impl);
	}

	decltype(auto) storage(auto &&... args) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) storage(auto &&... args) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
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
};

template <typename T, felt2::Dim D>
using InputGridTD = ByValue<T, D>;
using InputGrid = InputGridTD<felt2::Scalar, 3>;

template <typename T, felt2::Dim D, GridFlags flags = 0>
class ByRef
{
public:
	using This = ByRef<T, D, flags>;
	static constexpr bool is_device_shared = flags & GridFlag::is_device_shared;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using ContextImpl = host_or_device_context_t<flags>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByRef<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, ContextImpl, SizeImpl, StorageImpl>;

private:
	ref_if_child_t<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl;
	ActivateImpl m_activate_impl;
	AssertBoundsImpl const m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};

public:
	ByRef(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		VecDi const & offset_,
		Leaf background_) requires(is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_storage_impl{{context().context(), context().device()}},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, background_}
	{
		m_activate_impl.activate();
	}

	ByRef(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		VecDi const & offset_,
		Leaf background_) requires(!is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_storage_impl{},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, background_}
	{
		m_activate_impl.activate();
	}

//	ByRef(This const & other) noexcept
//		: m_context_impl{other.m_context_impl},
//		  m_size_impl{other.m_size_impl},
//		  m_storage_impl{other.m_storage_impl},
//		  m_activate_impl{
//			  m_context_impl, m_size_impl, m_storage_impl, other.m_activate_impl.m_background} {};

	ByRef(This && other) noexcept = default;

	ByRef(This const& other) noexcept = delete;

	~ByRef() = default;

	This & operator=(This && other) noexcept = default;
	This & operator=(This const & other) = delete;

	auto & context(this auto & self) noexcept
	{
		return unwrap_ref(self.m_context_impl);
	}
	decltype(auto) storage(auto &&... args) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) storage(auto &&... args) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
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
};

template <typename T, felt2::Dim D, GridFlags flags = 0>
class FilterTD
{
public:
	using This = FilterTD<T, D, flags>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::ResizableSize<Traits>;
	using StorageImpl = felt2::components::DataArraySpan<Traits>;
	using ContextImpl = host_or_device_context_t<flags>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	ref_if_child_t<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl{};
	AssertBoundsImpl m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	MatrixImpl m_matrix_impl{m_storage_impl};

public:
	FilterTD(decltype(m_context_impl) context_, SizeImpl size_impl)
		: m_context_impl{std::move(context_)}, m_size_impl{std::move(size_impl)}
	{
	}

	// Note: reference semantics mean we can't blindly copy `other` - reference_wrappers for e.g.
	// storage would still point to the original instance.
	FilterTD(This const & other) noexcept
		: m_context_impl{other.m_context_impl},
		  m_size_impl{other.m_size_impl},
		  m_storage_impl{other.m_storage_impl},
		  m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl},
		  m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_storage_impl}
	{
	}

	FilterTD(This && other) noexcept = default;

	~FilterTD() = default;

	This & operator=(This && other) noexcept = default;
	This & operator=(This const & other) = delete;

	decltype(auto) storage(auto &&... args) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) storage(auto &&... args) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
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
	decltype(auto) set(auto &&... args) const noexcept
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

template <typename T, felt2::Dim D, GridFlags flags = 0>
class ConvGridTD
{
public:
	using This = ConvGridTD<T, D, flags>;
	static constexpr bool is_device_shared = flags & GridFlag::is_device_shared;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;
	using Child = FilterTD<Leaf, Traits::k_dims, flags | GridFlag::is_child>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, flags | GridFlag::is_child>;

	using ContextImpl = host_or_device_context_t<flags>;
	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using BytesImpl = felt2::components::StorageBytes<StorageImpl>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using MatrixImpl = felt2::components::MatrixColPerChild<Traits, StorageImpl, ChildrenSizeImpl>;

private:
	ref_if_child_t<ContextImpl, flags> m_context_impl;
	StorageImpl m_storage_impl;
	SizeImpl const m_size_impl;
	ChildrenSizeImpl const m_children_size_impl;
	BytesImpl m_bytes_impl{m_storage_impl};
	AssertBoundsImpl m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	MatrixImpl m_matrix_impl{m_storage_impl, m_children_size_impl};

	ChildrenGrid m_children;

public:
	ConvGridTD(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		felt2::VecDi<D - 1> const & child_window_)
		: ConvGridTD{std::move(context_), size_, window_to_size(child_window_, size_), {0, 0, 0}}
	{
	}

	ConvGridTD(decltype(m_context_impl) context_, const VecDi & size_, const VecDi & child_size_)
		: ConvGridTD{context_, size_, child_size_, {0, 0, 0}}
	{
		assert(
			size_(D - 1) == child_size_(D - 1) &&
			"Channel dimension must be equal for both image and filters");
	}

	ConvGridTD(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		felt2::VecDi<D - 1> const & child_window_) requires(is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_storage_impl{{context().context(), context().device()}},
		  m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, window_to_size(child_window_, m_size_impl.size())},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_context_impl,
			  m_storage_impl,
			  Child{m_context_impl, {VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
	}

	ConvGridTD(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		VecDi const & child_size_,
		VecDi const & offset_) requires(!is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_context_impl,
			  m_storage_impl,
			  Child{m_context_impl, {VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
	}

	// Note: if/when copying is required, cannot default due to reference copying.
	ConvGridTD(This const & other) noexcept = delete;

	ConvGridTD(This && other) noexcept = default;

	~ConvGridTD() = default;

	This & operator=(This && other) noexcept = default;
	This & operator=(This const & other) = delete;

	auto & context(this auto & self) noexcept
	{
		return unwrap_ref(self.m_context_impl);
	}

	const ChildrenGrid & children() const noexcept
	{
		return m_children;
	}

	ChildrenGrid & children() noexcept
	{
		return m_children;
	}

	decltype(auto) storage(auto &&... args) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) storage(auto &&... args) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) bytes(auto &&... args) noexcept
	{
		return m_bytes_impl.bytes(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) bytes(auto &&... args) const noexcept
	{
		return m_bytes_impl.bytes(std::forward<decltype(args)>(args)...);
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

template <typename T, felt2::Dim D, GridFlags flags = 0>
class TemplateParentGridTD
{
public:
	using This = TemplateParentGridTD<T, D, flags>;
	static constexpr bool is_device_shared = flags & GridFlag::is_device_shared;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using ContextImpl = host_or_device_context_t<flags>;
	using Child = FilterTD<Leaf, Traits::k_dims, flags | GridFlag::is_child>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, flags | GridFlag::is_child>;

	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;

private:
	ref_if_child_t<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	ChildrenSizeImpl m_children_size_impl;
	ChildrenGrid m_children;

public:
	TemplateParentGridTD(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const felt2::VecDi<D - 1> & child_window_)
		: TemplateParentGridTD{
			  std::move(context_), size_, window_to_size(child_window_, size_), {0, 0, 0}}
	{
	}

	TemplateParentGridTD(
		decltype(m_context_impl) context_, const VecDi & size_, const VecDi & child_size_)
		: TemplateParentGridTD{std::move(context_), size_, child_size_, {0, 0, 0}}
	{
		assert(
			size_(D - 1) == child_size_(D - 1) &&
			"Channel dimension must be equal for both image and filters");
	}

	TemplateParentGridTD(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const felt2::VecDi<D - 1> & child_window_) requires(is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, window_to_size(child_window_, m_size_impl.size())},
		  m_children{m_children_size_impl.template make_empty_children<decltype(m_children)>(
			  context(), Child{context(), {VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
	}

	TemplateParentGridTD(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const VecDi & child_size_,
		const VecDi & offset_) requires(!is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_empty_children<decltype(m_children)>(
			  Child{{VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
	}

	// Note: if/when copying is required, cannot default due to reference copying.
	TemplateParentGridTD(This const & other) noexcept = delete;

	TemplateParentGridTD(This && other) noexcept = default;

	~TemplateParentGridTD() = default;

	This & operator=(This && other) noexcept = default;
	This & operator=(This const & other) = delete;

	auto & context(this auto & self) noexcept
	{
		return unwrap_ref(self.m_context_impl);
	}

	auto & children(this auto & self) noexcept
	{
		return self.m_children;
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
}  // namespace convfelt