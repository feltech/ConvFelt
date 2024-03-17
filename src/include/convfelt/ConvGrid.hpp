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
class USMMatrix
{
	struct Traits
	{
		using Leaf = felt2::Scalar;
	};
	using StorageImpl = felt2::components::USMRawArray<Traits>;
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

	operator MatrixImpl::Matrix() { // NOLINT(*-explicit-constructor)
		return m_matrix_impl.matrix();
	}

private:

	StorageImpl m_storage_impl;
	BytesImpl m_bytes_impl;
	MatrixImpl const m_matrix_impl;
};

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
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using LogImpl = felt2::components::Log;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, LogImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, SizeImpl, StorageImpl, LogImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	SizeImpl const m_size_impl;
	LogImpl m_log_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	StorageImpl m_storage_impl;
	ActivateImpl m_activate_impl;
	MatrixImpl const m_matrix_impl{m_storage_impl};

public:
	ByValue(
		const VecDi & size,
		const VecDi & offset,
		Leaf background,
		sycl::context context,
		sycl::device device)
		requires(is_device_shared)
		: m_size_impl{size, offset},
		  m_storage_impl{{std::move(context), std::move(device)}},
		  m_activate_impl{m_size_impl, m_storage_impl, m_log_impl, background}
	{
		m_activate_impl.activate();
	}

	ByValue(const VecDi & size_, const VecDi & offset_, Leaf background_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_storage_impl{},
		  m_activate_impl{m_size_impl, m_storage_impl, m_log_impl, background_}
	{
		m_activate_impl.activate();
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
	decltype(auto) log(auto &&... args) const noexcept
	{
		return m_log_impl.log(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) text(auto &&... args) const noexcept
	{
		return m_log_impl.text(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) reset_log(auto &&... args) noexcept
	{
		return m_log_impl.reset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) has_logs(auto &&... args) const noexcept
	{
		return m_log_impl.has_logs(std::forward<decltype(args)>(args)...);
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
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::Size<Traits>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using LogImpl = felt2::components::Log;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, LogImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByRef<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using ActivateImpl = felt2::components::Activate<Traits, SizeImpl, StorageImpl, LogImpl>;

private:
	SizeImpl const m_size_impl;
	LogImpl m_log_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	StorageImpl m_storage_impl;
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
		  m_storage_impl{{std::move(context), std::move(device)}},
		  m_activate_impl{m_size_impl, m_storage_impl, m_log_impl, background_}
	{
		m_activate_impl.activate();
	}

	ByRef(const VecDi & size_, const VecDi & offset_, Leaf background_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_storage_impl{},
		  m_activate_impl{m_size_impl, m_storage_impl, m_log_impl, background_}
	{
		m_activate_impl.activate();
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
	decltype(auto) log(auto &&... args) const noexcept
	{
		return m_log_impl.log(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) text(auto &&... args) const noexcept
	{
		return m_log_impl.text(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) reset_log(auto &&... args) noexcept
	{
		return m_log_impl.reset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) has_logs(auto &&... args) const noexcept
	{
		return m_log_impl.has_logs(std::forward<decltype(args)>(args)...);
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
	using Leaf = typename Traits::Leaf;

	using SizeImpl = felt2::components::ResizableSize<Traits>;
	using StorageImpl = felt2::components::DataArraySpan<Traits>;
	using LogImpl = felt2::components::Log;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, LogImpl, SizeImpl, StorageImpl>;
	using AccessImpl =
		felt2::components::AccessByValue<Traits, SizeImpl, StorageImpl, AssertBoundsImpl>;
	using MatrixImpl = felt2::components::EigenMap<Traits, StorageImpl>;

private:
	SizeImpl m_size_impl;
	StorageImpl m_storage_impl{};
	LogImpl m_log_impl{};
	AssertBoundsImpl const m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl};
	AccessImpl m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl};
	MatrixImpl const m_matrix_impl{m_storage_impl};

public:
	explicit FilterTD(SizeImpl size_impl) : m_size_impl{std::move(size_impl)} {}

	FilterTD(This const & other)
		: m_size_impl{other.m_size_impl},
		  m_storage_impl{other.m_storage_impl},
		  m_log_impl{other.m_log_impl},
		  m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl},
		  m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_storage_impl}
	{
	}

	FilterTD(This && other) noexcept
		: m_size_impl{std::move(other.m_size_impl)},
		  m_storage_impl{std::move(other.m_storage_impl)},
		  m_log_impl{std::move(other.m_log_impl)},
		  m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl},
		  m_access_impl{m_size_impl, m_storage_impl, m_assert_bounds_impl},
		  m_matrix_impl{m_storage_impl}
	{
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
	using Leaf = typename Traits::Leaf;
	using Child = FilterTD<Leaf, Traits::k_dims>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, is_device_shared>;

	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;
	using StorageImpl = std::conditional_t<
		is_device_shared,
		felt2::components::USMResizeableArray<Traits>,
		felt2::components::DataArray<Traits>>;
	using BytesImpl = felt2::components::StorageBytes<StorageImpl>;
	using LogImpl = felt2::components::Log;
	using AssertBoundsImpl =
		felt2::components::AssertBounds<Traits, LogImpl, SizeImpl, StorageImpl>;
	using MatrixImpl = felt2::components::MatrixColPerChild<Traits, StorageImpl, ChildrenSizeImpl>;

private:
	StorageImpl m_storage_impl;
	SizeImpl const m_size_impl;
	ChildrenSizeImpl const m_children_size_impl;
	LogImpl m_log_impl{};
	BytesImpl m_bytes_impl{m_storage_impl};
	AssertBoundsImpl const m_assert_bounds_impl{m_log_impl, m_size_impl, m_storage_impl};
	MatrixImpl m_matrix_impl{m_storage_impl, m_children_size_impl};

	ChildrenGrid m_children;

public:
	ConvGridTD(const VecDi & size_, const felt2::VecDi<D - 1> & child_window_)
		: ConvGridTD{size_, window_to_size(child_window_, size_), {0, 0, 0}}
	{
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_)
		: ConvGridTD{size_, child_size_, {0, 0, 0}}
	{
		assert(
			size_(D - 1) == child_size_(D - 1) &&
			"Channel dimension must be equal for both image and filters");
	}

	ConvGridTD(
		const VecDi & size_,
		const felt2::VecDi<D - 1> & child_window_,
		sycl::context const & context,
		sycl::device const & device)
		requires(is_device_shared)
		: m_storage_impl{{context, device}},
		  m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, window_to_size(child_window_, m_size_impl.size())},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_storage_impl, Child{{VecDi::Zero(), VecDi::Zero()}}, context, device)}
	{
		assert_child_size();
	}

	ConvGridTD(const VecDi & size_, const VecDi & child_size_, const VecDi & offset_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_children_span<decltype(m_children)>(
			  m_storage_impl, Child{{VecDi::Zero(), VecDi::Zero()}})}
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
	decltype(auto) log(auto &&... args) const noexcept
	{
		return m_log_impl.log(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) text(auto &&... args) const noexcept
	{
		return m_log_impl.text(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) reset_log(auto &&... args) noexcept
	{
		return m_log_impl.reset(std::forward<decltype(args)>(args)...);
	}
	decltype(auto) has_logs(auto &&... args) const noexcept
	{
		return m_log_impl.has_logs(std::forward<decltype(args)>(args)...);
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

template <typename T, felt2::Dim D, bool is_device_shared = false>
class TemplateParentGridTD
{
public:
	using This = TemplateParentGridTD<T, D>;

	struct Traits
	{
		using Leaf = T;
		static constexpr felt2::Dim k_dims = D;
	};

	using VecDi = felt2::VecDi<Traits::k_dims>;
	using Leaf = typename Traits::Leaf;
	using Child = FilterTD<Leaf, Traits::k_dims>;
	using ChildrenGrid = ByRef<Child, Traits::k_dims, is_device_shared>;

	using SizeImpl = felt2::components::Size<Traits>;
	using ChildrenSizeImpl = felt2::components::ChildrenSize<Traits, SizeImpl>;

private:
	SizeImpl const m_size_impl;
	ChildrenSizeImpl const m_children_size_impl;
	ChildrenGrid m_children;

public:
	TemplateParentGridTD(const VecDi & size_, const felt2::VecDi<D - 1> & child_window_)
		: TemplateParentGridTD{size_, window_to_size(child_window_, size_), {0, 0, 0}}
	{
	}

	TemplateParentGridTD(const VecDi & size_, const VecDi & child_size_)
		: TemplateParentGridTD{size_, child_size_, {0, 0, 0}}
	{
		assert(
			size_(D - 1) == child_size_(D - 1) &&
			"Channel dimension must be equal for both image and filters");
	}

	TemplateParentGridTD(
		const VecDi & size_,
		const felt2::VecDi<D - 1> & child_window_,
		sycl::context const & context,
		sycl::device const & device)
		requires(is_device_shared)
		: m_size_impl{size_, {0, 0, 0}},
		  m_children_size_impl{m_size_impl, window_to_size(child_window_, m_size_impl.size())},
		  m_children{m_children_size_impl.template make_empty_children<decltype(m_children)>(
			  Child{{VecDi::Zero(), VecDi::Zero()}}, context, device)}
	{
		assert_child_size();
	}

	TemplateParentGridTD(const VecDi & size_, const VecDi & child_size_, const VecDi & offset_)
		requires(!is_device_shared)
		: m_size_impl{size_, offset_},
		  m_children_size_impl{m_size_impl, child_size_},
		  m_children{m_children_size_impl.template make_empty_children<decltype(m_children)>(
			  Child{{VecDi::Zero(), VecDi::Zero()}})}
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