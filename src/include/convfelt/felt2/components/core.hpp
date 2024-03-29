// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/repeat_n.hpp>

#include "../typedefs.hpp"

#include "../index.hpp"

namespace felt2::components
{

constexpr auto unwrap(auto && val_)
{
	return static_cast<std::unwrap_reference_t<std::decay_t<decltype(val_)>>>(
		std::forward<decltype(val_)>(val_));
}

template <class T>
concept IsStream = requires(T obj_)
{
	{obj_ << "char array"};
	{obj_ << std::size_t{}};
	{obj_ << int{}};
	{obj_ << float{}};
	{obj_ << double{}};
};

template <class T>
concept HasDims = requires(T)
{
	{
		T::k_dims
	} -> std::convertible_to<Dim>;
};

template <class T>
concept HasLeafType = requires(T)
{
	typename T::Leaf;
};

template <typename T>
concept HasDimsAndLeafType = HasDims<T> && HasLeafType<T>;

template <class T>
concept IsPointer = std::is_pointer_v<T>;

template <class T>
concept IsStorage = requires(T obj_)
{
	{
		obj_.data()
	} -> IsPointer;
	{
		obj_.size()
	} -> std::convertible_to<PosIdx>;
};

template <typename T>
using ValueTypeT = std::remove_pointer_t<std::decay_t<T>>;

template <class T>
concept IsResizeableStorage = IsStorage<T> && requires(T obj_)
{
	{obj_.reserve(std::declval<PosIdx>())};
	{obj_.push_back(std::declval<ValueTypeT<decltype(obj_.data())>>())};
};

template <class T>
concept IsResizeable = requires(T obj_)
{
	{obj_.resize(std::declval<PosIdx>())};
};

template <class T>
concept HasStorageType = IsStorage<typename T::DataArray>;

template <class T>
concept HasStorage = requires(T obj_)
{
	{
		obj_.storage()
	} -> IsStorage;
};

template <HasStorage Storage>
using StorageLeafT =
	std::remove_pointer_t<std::decay_t<decltype(std::declval<Storage>().storage().data())>>;

template <class T>
concept HasResizeableStorage = requires(T obj_)
{
	{
		obj_.storage()
	} -> IsResizeableStorage;
};

template <class T>
concept HasSpanStorage = requires(T obj_)
{
	{
		obj_.storage().subspan(std::declval<std::size_t>(), std::declval<std::size_t>())
	} -> std::same_as<std::span<typename std::decay_t<decltype(obj_.storage())>::value_type>>;
};

template <class T>
concept HasBytes = requires(T obj_)
{
	{
		obj_.bytes()
	} -> std::same_as<std::span<std::byte>>;
};

template <class T>
concept HasStream = requires(T obj_)
{
	{
		obj_.has_stream()
	} -> std::convertible_to<bool>;
	{
		obj_.get_stream()
	} -> IsStream;
};

template <class T>
concept HasLog = requires(T obj_)
{
	{
		obj_.log()
	} -> std::convertible_to<bool>;
};

template <class T>
concept HasAbort = requires(T obj_)
{
	{obj_.abort()};
};

template <class T>
concept IsContext = requires(T obj_)
{
	requires HasLog<decltype(obj_.logger())>;
	requires HasAbort<decltype(obj_.aborter())>;
};

template <class T>
concept HasSize = requires(T obj_)
{
	typename T::VecDi;
	{
		obj_.size()
	} -> std::convertible_to<typename T::VecDi>;
	{
		obj_.offset()
	} -> std::convertible_to<typename T::VecDi>;
	{
		obj_.index(std::declval<typename T::VecDi>())
	} -> std::convertible_to<PosIdx>;
	{
		obj_.index(std::declval<PosIdx>())
	} -> std::convertible_to<typename T::VecDi>;
};

template <class T>
concept HasSizeCheck = HasSize<T> && requires(T obj_)
{
	{
		obj_.inside(std::declval<typename T::VecDi>())
	} -> std::convertible_to<bool>;
};

template <class T>
concept HasResize = requires(T obj_)
{
	typename T::VecDi;
	{obj_.resize(std::declval<typename T::VecDi>(), std::declval<typename T::VecDi>())};
};

template <class T, Dim D>
concept HasAssertBounds = requires(T obj_)
{
	{obj_.assert_pos_bounds(std::declval<PosIdx>(), std::declval<char const *>())};

	{obj_.assert_pos_idx_bounds(std::declval<VecDi<D>>(), std::declval<char const *>())};

	{obj_.assert_pos_bounds(std::declval<VecDi<D>>(), std::declval<char const *>())};

	{obj_.assert_pos_idx_bounds(std::declval<PosIdx>(), std::declval<char const *>())};
};

template <class T>
concept HasReadAccess = requires(T obj_)
{
	typename T::VecDi;
	typename T::Leaf;
	{
		obj_.get(std::declval<typename T::VecDi>())
	} -> std::convertible_to<typename T::Leaf const &>;
	{
		obj_.get(std::declval<PosIdx>())
	} -> std::convertible_to<typename T::Leaf const &>;
};
template <class T>
concept IsGrid = HasLeafType<T> && HasStorage<T> && HasSize<T> && HasReadAccess<T>;

template <class T>
concept IsSpanGrid = IsGrid<T> && HasResize<T> && HasSpanStorage<T>;

template <class T>
concept IsGridOfSpanGrids = IsGrid<T> && requires
{
	typename T::Leaf;
	IsSpanGrid<typename T::Leaf>;
};

template <class T>
concept HasChildrenSize = requires(T obj_)
{
	typename T::VecDi;
	{
		obj_.num_elems_per_child()
	} -> std::same_as<PosIdx>;
	{
		obj_.num_children()
	} -> std::same_as<PosIdx>;
};

template <typename T, Dim D>
void format_pos(IsStream auto & stream_, VecDT<T, D> const & pos_)
{
	stream_ << "(" << pos_(0);
	for (Dim axis = 1; axis < pos_.size(); ++axis) stream_ << ", " << pos_(axis);
	stream_ << ")";
}

template <HasLog Logger, HasAbort Aborter>
struct Context
{
	Logger m_logger_impl;
	Aborter m_aborter_impl;

	auto & logger(this auto & self_)
	{
		return self_.m_logger_impl;
	}
	auto & aborter(this auto & self_)
	{
		return self_.m_aborter_impl;
	}
};

template <HasDims Traits>
struct Size
{
	/// Dimension of the grid.
	static constexpr Dim k_dims = Traits::k_dims;
	/// D-dimensional signed integer vector.
	using VecDi = VecDi<k_dims>;

	/// The dimensions (size) of the grid.
	VecDi m_size;
	/// The translational offset of the grid's zero coordinate.
	VecDi m_offset;
	/// Cache for use in `inside`.
	VecDi m_offset_plus_size{m_offset + m_size};

	[[nodiscard]] VecDi const & size() const noexcept
	{
		return m_size;
	}

	[[nodiscard]] VecDi const & offset() const noexcept
	{
		return m_offset;
	}

	/**
	 * Get index in data array of position vector.
	 *
	 * The grid is packed in a 1D array, so this method is required to
	 * get the index in that array of the D-dimensional position.
	 *
	 * @param pos_ position in grid to query.
	 * @return index in internal data array of this grid position.
	 */
	[[nodiscard]] PosIdx index(VecDi const & pos_) const noexcept
	{
		return felt2::index(pos_, size(), offset());
	}

	/**
	 * Get position of index.
	 *
	 * Given an index in the 1D grid data array, calculate the position vector that it pertains to.
	 *
	 * @param idx_ index in internal data array to query.
	 * @return the position in the grid represented in the data array at given index.
	 */
	[[nodiscard]] VecDi index(PosIdx const idx_) const noexcept
	{
		return felt2::index(idx_, size(), offset());
	}

	/**
	 * Test if a position is inside the grid bounds.
	 *
	 * @tparam T the type of position vector (i.e. float vs. int).
	 * @param pos_ position in grid to query.
	 * @return true if position lies inside the grid, false otherwise.
	 */
	template <typename T>
	[[nodiscard]] bool inside(VecDT<T, k_dims> const & pos_) const noexcept
	{
		return felt2::inside(pos_, m_offset, m_offset_plus_size);
	}

	void resize(VecDi const & size_, VecDi const & offset_) noexcept
	{
		m_size = size_;
		m_offset = offset_;
		m_offset_plus_size = m_offset + size_;
	}
};

template <HasDims Traits, IsContext Context, HasSizeCheck Size, HasStorage Storage>
struct AssertBounds
{
	static constexpr Dim k_dims = Traits::k_dims;
	using VecDi = VecDi<k_dims>;

	std::reference_wrapper<Context> m_context_impl;
	std::reference_wrapper<Size const> m_size_impl;
	std::reference_wrapper<Storage const> m_storage_impl;

	bool assert_pos_bounds(PosIdx const pos_idx_, char const * title_) const
	{
		return assert_pos_idx_bounds(pos_idx_, title_);
	}

	bool assert_pos_idx_bounds(VecDi const & pos_, char const * title_) const
	{
		return assert_pos_bounds(pos_, title_);
	}

	bool assert_pos_bounds(VecDi const & pos_, char const * title_) const
	{
		if (!m_size_impl.get().inside(pos_))
		{
			typename Size::VecDi const k_max_extent =
				m_size_impl.get().offset() + m_size_impl.get().size();
			m_context_impl.get().logger().log(
				"AssertionError: ",
				title_,
				" assert_pos_bounds",
				pos_,
				" not in ",
				m_size_impl.get().offset(),
				" - ",
				k_max_extent,
				"\n");

			m_context_impl.get().aborter().abort();
			return false;
		}
		return true;
	}

	bool assert_pos_idx_bounds(PosIdx const pos_idx_, char const * title_) const
	{
		if (pos_idx_ >= m_storage_impl.get().storage().size())
		{
			VecDi pos;
			if (m_storage_impl.get().storage().size() > 0)
			{
				pos = m_size_impl.get().index(pos_idx_);
			}
			else
			{
				constexpr auto k_nan = std::numeric_limits<typename VecDi::Scalar>::quiet_NaN();
				pos = VecDi::Constant(k_nan);
			}

			m_context_impl.get().logger().log(
				"AssertionError: ",
				title_,
				" assert_pos_idx_bounds(",
				pos_idx_,
				") i.e. ",
				pos,
				" is greater than extent ",
				m_storage_impl.get().storage().size(),
				"\n");

			m_context_impl.get().aborter().abort();
			return false;
		}
		return true;
	}
};

template <HasDimsAndLeafType Traits, IsContext Context, HasSize Size, HasResizeableStorage Storage>
struct Activate
{
	/// Dimension of the grid.
	static Dim const k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;

	/// D-dimensional signed integer vector.
	using VecDi = VecDi<k_dims>;

	std::reference_wrapper<Context> m_context_impl;
	std::reference_wrapper<Size const> m_size_impl;
	std::reference_wrapper<Storage> m_storage_impl;
	Leaf m_background;

	/**
	 * Get whether this grid has been activated (data allocated) or not.
	 *
	 * @return true if data allocated, false if not.
	 */
	[[nodiscard]] bool is_active() const noexcept
	{
		return bool(m_storage_impl.get().storage().size());
	}

	/**
	 * Get the background value used to initially fill the grid.
	 *
	 * @return background value.
	 */
	[[nodiscard]] Leaf const & background() const noexcept
	{
		return m_background;
	}

	/**
	 * Construct the internal data array, initialising nodes to the background value.
	 */
	void activate()
	{
		//		assert(m_storage_impl.get().storage().size() == 0);
		// Note: resize() on libstdc++ 11 invokes operator=(), i.e. Copy/MoveAssignable, when we
		// only want to enforce Copy/MoveInsertable, i.e. copy/move constructor. So here we
		// essentially reimplement resize() (with the added precondition that data is empty).
		auto const new_size = static_cast<PosIdx>(m_size_impl.get().size().prod());
		m_storage_impl.get().storage().reserve(new_size);
		for (PosIdx idx = 0; idx < new_size; ++idx)
			m_storage_impl.get().storage().push_back(m_background);
	}

	/**
	 * Throw exception if grid is inactive
	 *
	 * @param title_ text to prefix exception message with.
	 */
	void assert_is_active(char const * title_) const noexcept
	{
		if (!is_active())
		{
			VecDi const & pos_min = m_size_impl.get().offset();
			VecDi const & pos_max = (m_size_impl.get().size() + pos_min - VecDi::Constant(1));
			m_context_impl.get().logger().log(
				title_, ": inactive grid ", pos_min, "-", pos_max, "\n");
			m_context_impl.get().aborter().abort();
		}
	}
};

template <
	HasDimsAndLeafType Traits,
	HasSize Size,
	HasStorage Storage,
	HasAssertBounds<Traits::k_dims> Assert>
struct AccessByRef
{
	/// Dimension of the grid.
	static Dim const k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// D-dimensional signed integer vector.
	using VecDi = felt2::VecDi<k_dims>;

	std::reference_wrapper<Size const> m_size_impl;
	std::reference_wrapper<Storage> m_storage_impl;
	std::reference_wrapper<Assert const> m_assert_impl;

	/**
	 * Get a reference to the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	Leaf & get(VecDi const & pos_) noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.get().assert_pos_bounds(pos_, "get: ");
#endif
		PosIdx const idx = m_size_impl.get().index(pos_);
		return get(idx);
	}

	/**
	 * Get a const reference to the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	[[nodiscard]] Leaf const & get(VecDi const & pos_) const noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.get().assert_pos_bounds(pos_, "get: ");
#endif
		PosIdx const idx = m_size_impl.get().index(pos_);
		return get(idx);
	}

	/**
	 * Get a reference to the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	Leaf & get(PosIdx const pos_idx_) noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.get().assert_pos_idx_bounds(pos_idx_, "get: ");
#endif
		return m_storage_impl.get().storage()[pos_idx_];
	}

	/**
	 * Get a const reference to the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	[[nodiscard]] Leaf const & get(PosIdx const pos_idx_) const noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.get().assert_pos_idx_bounds(pos_idx_, "get: ");
#endif
		return m_storage_impl.get().storage()[pos_idx_];
	}
};

template <
	HasDimsAndLeafType Traits,
	HasSize Size,
	HasStorage Storage,
	HasAssertBounds<Traits::k_dims> Assert>
struct AccessByValue
{
	/// Dimension of the grid.
	static constexpr Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// D-dimensional signed integer vector.
	using VecDi = felt2::VecDi<k_dims>;

	std::reference_wrapper<Size const> m_size_impl;
	std::reference_wrapper<Storage> m_storage_impl;
	std::reference_wrapper<Assert const> m_assert_impl;

	/**
	 * Get the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	[[nodiscard]] Leaf get(VecDi const & pos_) const noexcept
	{
		FELT2_DEBUG_CALL(m_assert_impl).get().assert_pos_bounds(pos_, "get: ");
		PosIdx const idx = m_size_impl.get().index(pos_);
		return get(idx);
	}

	/**
	 * Get the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	[[nodiscard]] Leaf get(PosIdx const pos_idx_) const noexcept
	{
		FELT2_DEBUG_CALL(m_assert_impl).get().assert_pos_idx_bounds(pos_idx_, "get: ");
		return m_storage_impl.get().storage()[pos_idx_];
	}

	/**
	 * Set the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @param val_ value to copy into grid at pos_.
	 */
	void set(VecDi const & pos_, Leaf val_) const noexcept
	{
		FELT2_DEBUG_CALL(m_assert_impl).get().assert_pos_bounds(pos_, "set: ");
		PosIdx const idx = m_size_impl.get().index(pos_);
		set(idx, val_);
	}

	/**
	 * Set the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @param val_ value to copy into grid at pos_.
	 */
	void set(PosIdx const pos_idx_, Leaf val_) const
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.get().assert_pos_bounds(pos_idx_, "set: ");
#endif
		m_storage_impl.get().storage()[pos_idx_] = val_;
	}
};

template <HasStorage Storage>
struct StorageBytes
{
	using Leaf = StorageLeafT<Storage>;

	std::reference_wrapper<Storage> m_storage_impl;

	[[nodiscard]] constexpr std::span<std::byte const> bytes() const noexcept
	{
		return std::as_bytes(m_storage_impl.get().storage());
	}

	[[nodiscard]] constexpr std::span<std::byte> bytes() noexcept
	{
		return std::as_writable_bytes(m_storage_impl.get().storage());
	}
};

template <HasLeafType Traits>
struct DataArray
{
	using Array = std::vector<typename Traits::Leaf>;

	Array m_data{};

	[[nodiscard]] constexpr Array & storage() noexcept
	{
		return m_data;
	}

	[[nodiscard]] constexpr Array const & storage() const noexcept
	{
		return m_data;
	}
};

template <HasLeafType Traits>
struct DataArraySpan
{
	using Array = std::span<typename Traits::Leaf>;

	Array m_data{};

	constexpr Array & storage() noexcept
	{
		return m_data;
	}

	[[nodiscard]] constexpr Array const & storage() const noexcept
	{
		return m_data;
	}
};

template <HasDims Traits, HasSize Size>
struct ChildrenSize
{
	static constexpr PosIdx k_dims = Traits::k_dims;
	using VecDi = felt2::VecDi<k_dims>;

	std::reference_wrapper<Size const> m_size_impl;
	/// Size of a child sub-grid.
	VecDi m_child_size;
	VecDi m_child_offset{(m_size_impl.get().offset().array() / m_child_size.array()).matrix()};
	VecDi m_children_size{
		[&]() constexpr noexcept
		{
			VecDi children_size =
				(m_size_impl.get().size().array() / m_child_size.array()).matrix();
			using Idx = typename VecDi::Index;

			// Ensure total size is covered with partitions in the case that total size doesn't
			// divide exactly.
			for (Idx dim = 0; dim < static_cast<Idx>(k_dims); ++dim)
				if (children_size(dim) * m_child_size(dim) != m_size_impl.get().size()(dim))
					children_size(dim) += 1;

			return children_size;
		}()};
	PosIdx m_num_children{static_cast<PosIdx>(m_children_size.prod())};
	PosIdx m_num_elems_per_child{static_cast<PosIdx>(m_child_size.prod())};

	/**
	 * Get size of child sub-grids.
	 *
	 * @return size of child sub-grid.
	 */
	[[nodiscard]] constexpr VecDi const & child_size() const noexcept
	{
		return m_child_size;
	}

	[[nodiscard]] constexpr VecDi const & child_offset() const noexcept
	{
		return m_child_offset;
	}

	[[nodiscard]] constexpr VecDi const & children_size() const noexcept
	{
		return m_children_size;
	}

	[[nodiscard]] constexpr PosIdx num_children() const noexcept
	{
		return m_num_children;
	}

	[[nodiscard]] constexpr PosIdx num_elems_per_child() const noexcept
	{
		return m_num_elems_per_child;
	}
	/**
	 * Calculate the position of a child grid (i.e. partition) given the position of leaf grid node.
	 *
	 * @param pos_leaf_ leaf grid node position vector.
	 * @return position index of spatial partition in which leaf position lies.
	 */
	[[nodiscard]] constexpr PosIdx pos_idx_child(VecDi const & pos_leaf_) const noexcept
	{
		// Encode child position as an index.
		return m_size_impl.index(pos_child(pos_leaf_));
	}

	/**
	 * Calculate the position of a child grid (i.e. partition) given the position of leaf grid node.
	 *
	 * @param pos_leaf_ leaf grid node position vector.
	 * @return position vector of spatial partition in which leaf position lies.
	 */
	[[nodiscard]] constexpr VecDi pos_child(VecDi const & pos_leaf_) const noexcept
	{
		// Position of leaf, without offset.
		auto pos_leaf_offset = pos_leaf_ - m_size_impl.offset();
		// Position of child grid containing leaf, without offset.
		auto pos_child_offset = (pos_leaf_offset.array() / m_child_size.array()).matrix();
		// Position of child grid containing leaf, including offset.
		auto pos_child = pos_child_offset + m_child_offset;
		return pos_child;
	}

	template <IsGridOfSpanGrids Children>
	[[nodiscard]] constexpr Children make_children_span(
		IsContext auto & context_,
		HasResizeableStorage auto & storage_impl_,
		IsSpanGrid auto background_) const
	{
		storage_impl_.storage().resize(m_num_elems_per_child * m_num_children);
		std::span const parent_data{storage_impl_.storage()};

		Children children{
			context_, m_children_size, m_size_impl.get().offset(), std::move(background_)};

		// Set each child sub-grid's size and offset.
		for (PosIdx pos_child_idx = 0; pos_child_idx < children.storage().size(); pos_child_idx++)
		{
			// Position of child in children grid.
			VecDi const pos_child_in_parent_with_offset = children.index(pos_child_idx);
			// Position of child in children grid, without offset.
			auto const pos_child_in_parent = pos_child_in_parent_with_offset - children.offset();
			// Scaled position of child == position in world space, without offset.
			auto const pos_child_in_world_without_parent_offset =
				(pos_child_in_parent.array() * m_child_size.array()).matrix();
			// Position of child in world space, including offset.
			auto const pos_child_in_world =
				pos_child_in_world_without_parent_offset + m_size_impl.get().offset();

			// Calculate overflow at edge of grid.
			auto const pos_lower =
				(pos_child_in_parent_with_offset.array() * m_child_size.array()).matrix();
			auto const pos_upper = (pos_lower.array() + m_child_size.array()).matrix();
			auto const signed_overflow = pos_upper - m_size_impl.get().size();
			auto const overflow = signed_overflow.cwiseMax(0);

			auto & child = children.get(pos_child_idx);

			child.resize(m_child_size - overflow, pos_child_in_world);

			auto const num_used_child_idxs = static_cast<felt2::PosIdx>(child.size().prod());
			child.storage() =
				parent_data.subspan(pos_child_idx * m_num_elems_per_child, num_used_child_idxs);
		}

		return children;
	}

	template <IsGridOfSpanGrids Children>
	[[nodiscard]] constexpr Children make_empty_children(
		IsContext auto & context_, IsSpanGrid auto background_) const
	{
		Children children{
			context_, m_children_size, m_size_impl.get().offset(), std::move(background_)};

		// Set each child sub-grid's size and offset.
		for (PosIdx pos_child_idx = 0; pos_child_idx < children.storage().size(); pos_child_idx++)
		{
			// Position of child in children grid.
			VecDi const pos_child_in_parent_with_offset = children.index(pos_child_idx);
			// Position of child in children grid, without offset.
			auto const pos_child_in_parent = pos_child_in_parent_with_offset - children.offset();
			// Scaled position of child == position in world space, without offset.
			auto const pos_child_in_world_without_parent_offset =
				(pos_child_in_parent.array() * m_child_size.array()).matrix();
			// Position of child in world space, including offset.
			auto const pos_child_in_world =
				pos_child_in_world_without_parent_offset + m_size_impl.get().offset();

			// Calculate overflow at edge of grid.
			auto const pos_lower =
				(pos_child_in_parent_with_offset.array() * m_child_size.array()).matrix();
			auto const pos_upper = (pos_lower.array() + m_child_size.array()).matrix();
			auto const signed_overflow = pos_upper - m_size_impl.get().size();
			auto const overflow = signed_overflow.cwiseMax(0);

			auto & child = children.get(pos_child_idx);

			child.resize(m_child_size - overflow, pos_child_in_world);
		}

		return children;
	}
};

struct Log
{
	template <typename... Args>
	bool log(Args &&... args_) const noexcept  // NOLINT(modernize-use-nodiscard)
	{
		fmt::print("{}", fmt::join(std::tuple{std::forward<Args>(args_)...}, ""));
		return true;
	}
};

struct Aborter
{
	static void abort() noexcept
	{
		std::abort();
	}
};

}  // namespace felt2::components