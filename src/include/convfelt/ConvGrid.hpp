#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "felt2/typedefs.hpp"

#include "felt2/components/core.hpp"
#include "felt2/components/eigen.hpp"
#include "felt2/components/sycl.hpp"

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
		is_device_shared = 1U << 0U,
		is_child = 1U << 1U
	};
};

template <GridFlags flags>
using HostOrDeviceContextT = std::conditional_t<
	static_cast<bool>(flags & GridFlag::is_device_shared),
	// device grid
	felt2::components::device::
		Context<felt2::components::device::Log, felt2::components::device::Aborter>,
	// host grid
	felt2::components::Context<felt2::components::Log, felt2::components::Aborter>>;

template <class T, GridFlags flags>
using RefIfChildT =
	std::conditional_t<static_cast<bool>(flags & GridFlag::is_child), std::reference_wrapper<T>, T>;

constexpr decltype(auto) unwrap_ref(auto && maybe_wrapped_)
{
	return static_cast<std::add_lvalue_reference_t<
		std::unwrap_reference_t<std::remove_reference_t<decltype(maybe_wrapped_)>>>>(
		maybe_wrapped_);
}

inline auto make_host_context()
{
	return felt2::components::Context{felt2::components::Log{}, felt2::components::Aborter{}};
}

inline auto make_device_context(sycl::device const & dev_, sycl::context const & ctx_)
{
	return felt2::components::device::Context{
		felt2::components::device::Log{}, felt2::components::device::Aborter{}, dev_, ctx_};
}

inline auto make_device_context(sycl::queue const & queue_)
{
	return make_device_context(queue_.get_device(), queue_.get_context());
}

class USMMatrix
{
	struct Traits
	{
		using Leaf = felt2::Scalar;
	};
	using StorageImpl = felt2::components::device::USMRawArray<Traits>;
	using MatrixImpl = felt2::components::MatrixMap<StorageImpl>;

public:
	USMMatrix(
		felt2::Dim const rows_,
		felt2::Dim const cols_,
		sycl::device const & dev_,
		sycl::context const & ctx_)
		: m_storage_impl{static_cast<std::size_t>(rows_ * cols_), dev_, ctx_},
		  m_matrix_impl{rows_, cols_, m_storage_impl}
	{
	}

	[[nodiscard]] decltype(auto) matrix(auto &&... args_) noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args_)>(args_)...);
	}

	[[nodiscard]] decltype(auto) matrix(auto &&... args_) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args_)>(args_)...);
	}

	[[nodiscard]] decltype(auto) bytes() noexcept
	{
		return std::as_writable_bytes(m_storage_impl.storage());
	}

	[[nodiscard]] decltype(auto) bytes() const noexcept
	{
		return std::as_bytes(m_storage_impl.storage());
	}

	explicit operator MatrixImpl::Matrix()
	{  // NOLINT(*-explicit-constructor)
		return m_matrix_impl.matrix();
	}

private:
	StorageImpl m_storage_impl;
	MatrixImpl m_matrix_impl;
};

template <typename T, felt2::Dim D, GridFlags flags = 0>
class ByValue
{
public:
	using This = ByValue<T, D, flags>;
	static constexpr bool k_is_device_shared = (flags & GridFlag::is_device_shared) != 0;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using ContextImpl = HostOrDeviceContextT<flags>;
	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		k_is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	RefIfChildT<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl;
	ActivateImpl m_activate_impl;
	AssertBoundsImpl m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	MatrixImpl m_matrix_impl{m_storage_impl};

public:
	ByValue(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const VecDi & offset_,
		Leaf background_) requires(k_is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_storage_impl{{context().context(), context().device()}},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, std::move(background_)}
	{
		m_activate_impl.activate();
	}

	ByValue(
		decltype(m_context_impl) context_,
		const VecDi & size_,
		const VecDi & offset_,
		Leaf background_) requires(!k_is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_storage_impl{},
		  m_activate_impl{m_context_impl, m_size_impl, m_storage_impl, std::move(background_)}
	{
		m_activate_impl.activate();
	}

	// Note: if/when copying is required, cannot default due to reference copying.
	ByValue(This const & other_) noexcept = delete;

	ByValue(This && other_) noexcept = default;

	~ByValue() = default;

	This & operator=(This && other_) noexcept = default;
	This & operator=(This const & other_) = delete;

	auto & context(this auto & self_) noexcept
	{
		return unwrap_ref(self_.m_context_impl);
	}

	decltype(auto) storage(auto &&... args_) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) storage(auto &&... args_) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) index(auto &&... args_) const noexcept
	{
		return m_size_impl.index(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) get(VecDi const & pos_) noexcept
	{
		return m_access_impl.get(pos_);
	}
	[[nodiscard]] decltype(auto) get(VecDi const & pos_) const noexcept
	{
		return m_access_impl.get(pos_);
	}
	decltype(auto) get(auto &&... args_) noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) get(auto &&... args_) const noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) set(auto &&... args_) noexcept
	{
		return m_access_impl.set(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) offset(auto &&... args_) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) size(auto &&... args_) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) inside(auto &&... args_) const noexcept
	{
		return m_size_impl.inside(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) array(auto &&... args_) const noexcept
	{
		return m_matrix_impl.array(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) matrix(auto &&... args_) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args_)>(args_)...);
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
	static constexpr bool k_is_device_shared = (flags & GridFlag::is_device_shared) != 0;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		k_is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using ContextImpl = HostOrDeviceContextT<flags>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByRef<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, ContextImpl, SizeImpl, StorageImpl>;

private:
	RefIfChildT<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl;
	ActivateImpl m_activate_impl;
	AssertBoundsImpl m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};

public:
	ByRef(
		decltype(m_context_impl) context_,
		VecDi const & size_,
		VecDi const & offset_,
		Leaf background_) requires(k_is_device_shared)
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
		Leaf background_) requires(!k_is_device_shared)
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
	//			  m_context_impl, m_size_impl, m_storage_impl, other.m_activate_impl.m_background}
	//{};

	ByRef(This && other_) noexcept = default;

	ByRef(This const & other_) noexcept = delete;

	~ByRef() = default;

	This & operator=(This && other_) noexcept = default;
	This & operator=(This const & other_) = delete;

	auto & context(this auto & self_) noexcept
	{
		return unwrap_ref(self_.m_context_impl);
	}
	decltype(auto) storage(auto &&... args_) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) storage(auto &&... args_) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) index(auto &&... args_) const noexcept
	{
		return m_size_impl.index(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) get(VecDi const & pos_) noexcept
	{
		return m_access_impl.get(pos_);
	}
	[[nodiscard]] decltype(auto) get(VecDi const & pos_) const noexcept
	{
		return m_access_impl.get(pos_);
	}
	decltype(auto) get(auto &&... args_) noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) get(auto &&... args_) const noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) offset(auto &&... args_) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) size(auto &&... args_) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args_)>(args_)...);
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

	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = felt2::components::DataArraySpan<Traits>;
	using ContextImpl = HostOrDeviceContextT<flags>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	RefIfChildT<ContextImpl, flags> m_context_impl;
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl{};
	AssertBoundsImpl m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	MatrixImpl m_matrix_impl{m_storage_impl};

public:
	FilterTD(decltype(m_context_impl) context_, SizeImpl size_impl_)
		: m_context_impl{std::move(context_)}, m_size_impl{std::move(size_impl_)}
	{
	}

	// Note: reference semantics mean we can't blindly copy `other` - reference_wrappers for e.g.
	// storage would still point to the original instance.
	FilterTD(This const & other_) noexcept
		: m_context_impl{other_.m_context_impl},
		  m_size_impl{other_.m_size_impl},
		  m_storage_impl{other_.m_storage_impl},
		  m_assert_bounds_impl{m_context_impl, m_size_impl, m_storage_impl},
		  m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_storage_impl}
	{
	}

	FilterTD(This && other_) noexcept = default;

	~FilterTD() = default;

	This & operator=(This && other_) noexcept = default;
	This & operator=(This const & other_) = delete;

	decltype(auto) storage(auto &&... args_) noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) storage(auto &&... args_) const noexcept
	{
		return m_storage_impl.storage(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) index(auto &&... args_) const noexcept
	{
		return m_size_impl.index(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) get(VecDi const & pos_) noexcept
	{
		return m_access_impl.get(pos_);
	}
	[[nodiscard]] decltype(auto) get(VecDi const & pos_) const noexcept
	{
		return m_access_impl.get(pos_);
	}
	decltype(auto) get(auto &&... args_) noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] [[nodiscard]] decltype(auto) get(auto &&... args_) const noexcept
	{
		return m_access_impl.get(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) set(auto &&... args_) const noexcept
	{
		return m_access_impl.set(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) offset(auto &&... args_) const noexcept
	{
		return m_size_impl.offset(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) size(auto &&... args_) const noexcept
	{
		return m_size_impl.size(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) resize(auto &&... args_) noexcept
	{
		return m_size_impl.resize(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) inside(auto &&... args_) const noexcept
	{
		return m_size_impl.inside(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) array(auto &&... args_) const noexcept
	{
		return m_matrix_impl.array(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) matrix(auto &&... args_) const noexcept
	{
		return m_matrix_impl.matrix(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) assert_pos_bounds(auto &&... args_) const noexcept
	{
		return m_assert_bounds_impl.assert_pos_bounds(std::forward<decltype(args_)>(args_)...);
	}
	decltype(auto) assert_pos_idx_bounds(auto &&... args_) const noexcept
	{
		return m_assert_bounds_impl.assert_pos_idx_bounds(std::forward<decltype(args_)>(args_)...);
	}
};

using Filter = FilterTD<felt2::Scalar, 3>;

template <typename T, felt2::Dim D, GridFlags flags = 0>
class ConvGridTD
{
public:
	using This = ConvGridTD<T, D, flags>;
	static constexpr bool k_is_device_shared = (flags & GridFlag::is_device_shared) != 0;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;
	using Child = FilterTD<Leaf, Traits::k_dims, flags | GridFlag::is_child>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, flags | GridFlag::is_child>;

	using ContextImpl = HostOrDeviceContextT<flags>;
	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;
	using StorageImpl = std::conditional_t<
		k_is_device_shared,
		felt2::components::device::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, ContextImpl, SizeImpl, StorageImpl>;
	using MatrixImpl = felt2::components::MatrixColPerChild<Traits, StorageImpl, ChildrenSizeImpl>;

private:
	RefIfChildT<ContextImpl, flags> m_context_impl;
	StorageImpl m_storage_impl;
	SizeImpl m_size_impl;
	ChildrenSizeImpl m_children_size_impl;
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
		felt2::VecDi<D - 1> const & child_window_) requires(k_is_device_shared)
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
		VecDi const & offset_) requires(!k_is_device_shared)
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
	ConvGridTD(This const & other_) noexcept = delete;

	ConvGridTD(This && other_) noexcept = default;

	~ConvGridTD() = default;

	This & operator=(This && other_) noexcept = default;
	This & operator=(This const & other_) = delete;

	auto & context(this auto & self_) noexcept
	{
		return unwrap_ref(self_.m_context_impl);
	}

	[[nodiscard]] const ChildrenGrid & children() const noexcept
	{
		return m_children;
	}

	[[nodiscard]] ChildrenGrid & children() noexcept
	{
		return m_children;
	}

	[[nodiscard]] auto& storage(this auto && self_) noexcept
	{
		return self_.m_storage_impl.storage();
	}
	[[nodiscard]] auto bytes() noexcept
	{
		return std::as_writable_bytes(std::span{m_storage_impl.storage()});
	}
	[[nodiscard]] auto bytes() const noexcept
	{
		return std::as_bytes(std::span{m_storage_impl.storage()});
	}
	[[nodiscard]] auto const& offset() const noexcept
	{
		return m_size_impl.offset();
	}
	[[nodiscard]] auto const& size() const noexcept
	{
		return m_size_impl.size();
	}
	[[nodiscard]] auto const& child_size() const noexcept
	{
		return m_children_size_impl.child_size();
	}
	[[nodiscard]] auto inside(auto &&... args_) const noexcept
	{
		return m_size_impl.inside(std::forward<decltype(args_)>(args_)...);
	}
	[[nodiscard]] decltype(auto) matrix(this auto && self_) noexcept
	{
		return self_.m_matrix_impl.matrix();
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
		VecDi child_size;
		child_size << window_, size_(size_.size() - 1);
		return child_size;
	}
};

using ConvGrid = ConvGridTD<felt2::Scalar, 3>;

template <typename T, felt2::Dim D, GridFlags flags = 0>
class TemplateParentGridTD
{
public:
	using This = TemplateParentGridTD<T, D, flags>;
	static constexpr bool k_is_device_shared = (flags & GridFlag::is_device_shared) != 0;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;

	using ContextImpl = HostOrDeviceContextT<flags>;
	using Child = FilterTD<Leaf, Traits::k_dims, flags | GridFlag::is_child>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, flags | GridFlag::is_child>;

	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;

private:
	RefIfChildT<ContextImpl, flags> m_context_impl;
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
		const felt2::VecDi<D - 1> & child_window_) requires(k_is_device_shared)
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
		const VecDi & offset_) requires(!k_is_device_shared)
		: m_context_impl{std::move(context_)},
		  m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_empty_children<decltype(m_children)>(
			  Child{{VecDi::Zero(), VecDi::Zero()}})}
	{
		assert_child_size();
	}

	// Note: if/when copying is required, cannot default due to reference copying.
	TemplateParentGridTD(This const & other_) noexcept = delete;

	TemplateParentGridTD(This && other_) noexcept = default;

	~TemplateParentGridTD() = default;

	This & operator=(This && other_) noexcept = default;
	This & operator=(This const & other_) = delete;

	auto & context(this auto & self_) noexcept
	{
		return unwrap_ref(self_.m_context_impl);
	}

	auto & children(this auto & self_) noexcept
	{
		return self_.m_children;
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
		VecDi child_size;
		child_size << window_, size_(size_.size() - 1);
		return child_size;
	}
};
}  // namespace convfelt