// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <span>
#include <type_traits>
#include <vector>

#include "../typedefs.hpp"

#include "../index.hpp"

namespace felt2::components
{

template <class T>
concept IsStream = requires(T t) {
	{
		t << "char array"
	};
	{
		t << std::size_t{}
	};
	{
		t << int{}
	};
	{
		t << float{}
	};
	{
		t << double{}
	};
};

template <class T>
concept HasDims = requires(T) {
	{
		T::k_dims
	} -> std::convertible_to<Dim>;
};

template <class T>
concept HasLeafType = requires(T) { typename T::Leaf; };

template <typename T>
concept HasDimsAndLeafType = HasDims<T> && HasLeafType<T>;

template <class T>
concept IsPointer = std::is_pointer_v<T>;

template <class T>
concept IsStorage = requires(T obj) {
	{
		obj.data()
	} -> IsPointer;
	{
		obj.size()
	} -> std::convertible_to<PosIdx>;
};

template <typename T>
using value_type_t = std::remove_pointer_t<std::decay_t<T>>;

template <class T>
concept IsResizeableStorage = IsStorage<T> && requires(T obj) {
	{
		obj.reserve(std::declval<PosIdx>())
	};
	{
		obj.push_back(std::declval<value_type_t<decltype(obj.data())>>())
	};
};

template <class T>
concept IsResizeable = requires(T obj) {
	{
		obj.resize(std::declval<PosIdx>())
	};
};

template <class T>
concept HasStorageType = IsStorage<typename T::DataArray>;

template <class T>
concept HasStorage = requires(T obj) {
	{
		obj.storage()
	} -> IsStorage;
};

template <class T>
concept HasResizeableStorage = requires(T obj) {
	{
		obj.storage()
	} -> IsResizeableStorage;
};

template <class T>
concept HasStream = requires(T t) {
	{
		t.has_stream()
	} -> std::convertible_to<bool>;
	{
		t.get_stream()
	} -> IsStream;
};

template <class T>
concept HasSize = requires(T t) {
	typename T::VecDi;
	{
		t.size()
	} -> std::convertible_to<typename T::VecDi>;
	{
		t.offset()
	} -> std::convertible_to<typename T::VecDi>;
	{
		t.index(std::declval<typename T::VecDi>())
	} -> std::convertible_to<PosIdx>;
	{
		t.index(std::declval<PosIdx>())
	} -> std::convertible_to<typename T::VecDi>;
};

template <class T>
concept HasSizeCheck = HasSize<T> && requires(T t) {
	{
		t.inside(std::declval<typename T::VecDi>())
	} -> std::convertible_to<bool>;
};

template <class T>
concept HasResize = requires(T t) {
	typename T::VecDi;
	{
		t.resize(std::declval<typename T::VecDi>(), std::declval<typename T::VecDi>())
	};
};

template <class T, Dim D>
concept HasAssertBounds = requires(T t) {
	{
		t.assert_pos_bounds(std::declval<PosIdx>(), std::declval<const char *>())
	};

	{
		t.assert_pos_idx_bounds(std::declval<VecDi<D>>(), std::declval<const char *>())
	};

	{
		t.assert_pos_bounds(std::declval<VecDi<D>>(), std::declval<const char *>())
	};

	{
		t.assert_pos_idx_bounds(std::declval<PosIdx>(), std::declval<const char *>())
	};
};

template <class T>
concept HasReadAccess = requires(T t) {
	typename T::VecDi;
	typename T::Leaf;
	{
		t.get(std::declval<typename T::VecDi>())
	} -> std::convertible_to<const typename T::Leaf &>;
	{
		t.get(std::declval<PosIdx>())
	} -> std::convertible_to<const typename T::Leaf &>;
};
template <class T>
concept IsGrid = HasLeafType<T> && HasStorage<T> && HasSize<T> && HasReadAccess<T>;

template <class T>
concept IsSpanGrid = IsGrid<T> && HasResize<T> && HasResizeableStorage<T>;

template <class T>
concept IsGridOfSpanGrids = IsGrid<T> && requires {
	typename T::Leaf;
	IsSpanGrid<typename T::Leaf>;
};

template <class T>
concept HasChildrenSize = requires(T t) {
	typename T::VecDi;
	{
		t.num_elems_per_child()
	} -> std::same_as<PosIdx>;
	{
		t.num_children()
	} -> std::same_as<PosIdx>;
};

template <typename T, Dim D>
void format_pos(IsStream auto & stream, VecDT<T, D> const & pos)
{
	stream << "(" << pos(0);
	for (Dim axis = 1; axis < pos.size(); ++axis) stream << ", " << pos(axis);
	stream << ")";
}

template <HasDims Traits>
struct Size
{
	/// Dimension of the grid.
	static constexpr Dim k_dims = Traits::k_dims;
	/// D-dimensional signed integer vector.
	using VecDi = VecDi<k_dims>;

	/// The dimensions (size) of the grid.
	VecDi const m_size;
	/// The translational offset of the grid's zero coordinate.
	VecDi const m_offset;
	/// Cache for use in `inside`.
	VecDi const m_offset_plus_size{m_offset + m_size};

	const VecDi & size() const noexcept
	{
		return m_size;
	}

	const VecDi & offset() const noexcept
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
	PosIdx index(const VecDi & pos_) const noexcept
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
	VecDi index(const PosIdx idx_) const noexcept
	{
		return felt2::index(idx_, size(), offset());
	}

	/**
	 * Test if a position is inside the grid bounds.
	 *
	 * @tparam Pos the type of position vector (i.e. float vs. int).
	 * @param pos_ position in grid to query.
	 * @return true if position lies inside the grid, false otherwise.
	 */
	template <typename T>
	bool inside(const VecDT<T, k_dims> & pos_) const noexcept
	{
		return felt2::inside(pos_, m_offset, m_offset_plus_size);
	}
};

template <HasDims Traits>
struct ResizableSize
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

	const VecDi & size() const noexcept
	{
		return m_size;
	}

	VecDi & size() noexcept
	{
		return m_size;
	}

	const VecDi & offset() const noexcept
	{
		return m_offset;
	}

	VecDi & offset() noexcept
	{
		return m_offset;
	}

	void resize(const VecDi & size_, const VecDi & offset_) noexcept
	{
		m_size = size_;
		m_offset = offset_;
		m_offset_plus_size = m_offset + size_;
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
	PosIdx index(const VecDi & pos_) const noexcept
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
	VecDi index(const PosIdx idx_) const noexcept
	{
		return felt2::index(idx_, size(), offset());
	}

	/**
	 * Test if a position is inside the grid bounds.
	 *
	 * @tparam Pos the type of position vector (i.e. float vs. int).
	 * @param pos_ position in grid to query.
	 * @return true if position lies inside the grid, false otherwise.
	 */
	template <typename T>
	bool inside(const VecDT<T, k_dims> & pos_) const
	{
		return felt2::inside(pos_, m_offset, m_offset_plus_size);
	}
};

template <HasDims Traits, HasStream Stream, HasSizeCheck Size, HasStorage Storage>
struct AssertBounds
{
	static constexpr Dim k_dims = Traits::k_dims;
	using VecDi = VecDi<k_dims>;

	Stream & m_stream_impl;
	Size const & m_size_impl;
	Storage const & m_storage_impl;

	void assert_pos_bounds(const PosIdx pos_idx_, const char * title_) const
	{
		assert_pos_idx_bounds(pos_idx_, title_);
	}

	void assert_pos_idx_bounds(const VecDi & pos_, const char * title_) const
	{
		assert_pos_bounds(pos_, title_);
	}

	void assert_pos_bounds(const VecDi & pos_, const char * title_) const
	{
		if (!m_size_impl.inside(pos_))
		{
			if (!m_stream_impl.has_stream())
				assert(
					m_size_impl.inside(pos_) &&
					"assert_pos_bounds: debug unavailable, no stream set");

			m_stream_impl.get_stream() << "AssertionError: " << title_ << " assert_pos_bounds";
			format_pos(m_stream_impl.get_stream(), pos_);
			m_stream_impl.get_stream() << " is not in ";
			format_pos(m_stream_impl.get_stream(), m_size_impl.offset());
			typename Size::VecDi max_extent = m_size_impl.offset() + m_size_impl.size();
			m_stream_impl.get_stream() << " - ";
			format_pos(m_stream_impl.get_stream(), max_extent);
			m_stream_impl.get_stream() << "\n";
		}
		assert(m_size_impl.inside(pos_));
	}

	void assert_pos_idx_bounds(const PosIdx pos_idx_, const char * title_) const
	{
		if (pos_idx_ >= m_storage_impl.storage().size())
		{
			if (!m_stream_impl.has_stream())
				assert(
					pos_idx_ < m_storage_impl.storage().size() &&
					"assert_pos_idx_bounds: debug unavailable, no stream set");
			auto pos = m_size_impl.index(pos_idx_);
			m_stream_impl.get_stream() << "AssertionError: " << title_ << " assert_pos_idx_bounds("
									   << pos_idx_ << ") i.e. ";
			format_pos(m_stream_impl.get_stream(), pos);
			m_stream_impl.get_stream()
				<< " is greater than extent " << m_storage_impl.storage().size() << "\n";
		}
		assert(pos_idx_ < m_storage_impl.storage().size());
	}
};

template <HasDimsAndLeafType Traits, HasSize Size, HasResizeableStorage Storage, HasStream Stream>
struct Activate
{
	/// Dimension of the grid.
	static const Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;

	/// D-dimensional signed integer vector.
	using VecDi = VecDi<k_dims>;

	Size const & m_size_impl;
	Storage & m_storage_impl;
	Stream & m_stream_impl;
	Leaf const m_background;

	/**
	 * Get whether this grid has been activated (data allocated) or not.
	 *
	 * @return true if data allocated, false if not.
	 */
	[[nodiscard]] bool is_active() const noexcept
	{
		return bool(m_storage_impl.storage().size());
	}

	/**
	 * Get the background value used to initially fill the grid.
	 *
	 * @return background value.
	 */
	const Leaf & background() const noexcept
	{
		return m_background;
	}

	/**
	 * Construct the internal data array, initialising nodes to the background value.
	 */
	void activate()
	{
		assert(m_storage_impl.storage().size() == 0);
		// Note: resize() on libstdc++ 11 invokes operator=(), i.e. Copy/MoveAssignable, when we
		// only want to enforce Copy/MoveInsertable, i.e. copy/move constructor. So here we
		// essentially reimplement resize() (with the added precondition that data is empty).
		auto const new_size = static_cast<PosIdx>(m_size_impl.size().prod());
		m_storage_impl.storage().reserve(new_size);
		for (PosIdx idx = 0; idx < new_size; ++idx)
			m_storage_impl.storage().push_back(m_background);
	}

	/**
	 * Throw exception if grid is inactive
	 *
	 * @param title_ text to prefix exception message with.
	 */
	void assert_is_active(const char * title_) const noexcept
	{
		if (m_stream_impl.has_stream() && !is_active())
		{
			const VecDi & pos_min = m_size_impl.offset();
			const VecDi & pos_max = (m_size_impl.size() + pos_min - VecDi::Constant(1));
			m_stream_impl.get_stream() << title_ << ": inactive grid ";
			format_pos(m_stream_impl.get_stream(), pos_min);
			m_stream_impl.get_stream() << "-";
			format_pos(m_stream_impl.get_stream(), pos_max);
			m_stream_impl.get_stream() << ")";
		}
		assert(is_active());
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
	static const Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// D-dimensional signed integer vector.
	using VecDi = felt2::VecDi<k_dims>;

	Size const & m_size_impl;
	Storage & m_storage_impl;
	Assert const & m_assert_impl;

	/**
	 * Get a reference to the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	Leaf & get(const VecDi & pos_) noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_bounds(pos_, "get: ");
#endif
		const PosIdx idx = m_size_impl.index(pos_);
		return get(idx);
	}

	/**
	 * Get a const reference to the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	const Leaf & get(const VecDi & pos_) const noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_bounds(pos_, "get: ");
#endif
		const PosIdx idx = m_size_impl.index(pos_);
		return get(idx);
	}

	/**
	 * Get a reference to the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	Leaf & get(const PosIdx pos_idx_) noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_idx_bounds(pos_idx_, "get: ");
#endif
		return m_storage_impl.storage()[pos_idx_];
	}

	/**
	 * Get a const reference to the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	const Leaf & get(const PosIdx pos_idx_) const noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_idx_bounds(pos_idx_, "get: ");
#endif
		return m_storage_impl.storage()[pos_idx_];
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

	Size const & m_size_impl;
	Storage & m_storage_impl;
	Assert const & m_assert_impl;

	/**
	 * Get the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	Leaf get(const VecDi & pos_) const noexcept
	{
		FELT2_DEBUG_CALL(m_assert_impl).assert_pos_bounds(pos_, "get: ");
		const PosIdx idx = m_size_impl.index(pos_);
		return get(idx);
	}

	/**
	 * Get the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @return internally stored value at given grid position
	 */
	Leaf get(const PosIdx pos_idx_) const noexcept
	{
		FELT2_DEBUG_CALL(m_assert_impl).assert_pos_idx_bounds(pos_idx_, "get: ");
		return m_storage_impl.storage()[pos_idx_];
	}

	/**
	 * Set the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @param val_ value to copy into grid at pos_.
	 */
	void set(const VecDi & pos_, Leaf val_)
	{
		FELT2_DEBUG_CALL(m_assert_impl).assert_pos_bounds(pos_, "set: ");
		const PosIdx idx = m_size_impl.index(pos_);
		set(idx, val_);
	}

	/**
	 * Set the value stored in the grid.
	 *
	 * @param pos_idx_ data index of position to query.
	 * @param val_ value to copy into grid at pos_.
	 */
	void set(const PosIdx pos_idx_, Leaf val_)
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_bounds(pos_idx_, "set: ");
#endif
		m_storage_impl.storage()[pos_idx_] = val_;
	}
};

template <HasLeafType Traits>
struct DataArray
{
	using Array = std::vector<typename Traits::Leaf>;

	Array m_data{};

	Array & storage()
	{
		return m_data;
	}

	Array const & storage() const
	{
		return m_data;
	}
};

template <HasLeafType Traits>
struct DataArraySpan
{
	using Array = std::span<typename Traits::Leaf>;

	Array m_data{};

	Array & storage()
	{
		return m_data;
	}

	Array const & storage() const
	{
		return m_data;
	}
};

template <HasDims Traits, HasSize Size>
struct ChildrenSize
{
	static constexpr PosIdx k_dims = Traits::k_dims;
	using VecDi = felt2::VecDi<k_dims>;

	Size const & m_size_impl;
	/// Size of a child sub-grid.
	VecDi const m_child_size;
	VecDi const m_child_offset{(m_size_impl.offset().array() / m_child_size.array()).matrix()};
	VecDi const m_children_size{
		[&]() noexcept
		{
			VecDi children_size = (m_size_impl.size().array() / m_child_size.array()).matrix();
			using Idx = typename VecDi::Index;

			// Ensure total size is covered with partitions in the case that total size doesn't
			// divide exactly.
			for (Idx dim = 0; dim < static_cast<Idx>(k_dims); ++dim)
				if (children_size(dim) * m_child_size(dim) != m_size_impl.size()(dim))
					children_size(dim) += 1;

			return children_size;
		}()};
	PosIdx const m_num_children{static_cast<PosIdx>(m_children_size.prod())};
	PosIdx const m_num_elems_per_child{static_cast<PosIdx>(m_child_size.prod())};

	/**
	 * Get size of child sub-grids.
	 *
	 * @return size of child sub-grid.
	 */
	[[nodiscard]] const VecDi & child_size() const noexcept
	{
		return m_child_size;
	}

	[[nodiscard]] const VecDi & child_offset() const noexcept
	{
		return m_child_offset;
	}

	[[nodiscard]] const VecDi & children_size() const noexcept
	{
		return m_children_size;
	}

	[[nodiscard]] PosIdx num_children() const noexcept
	{
		return m_num_children;
	}

	[[nodiscard]] PosIdx num_elems_per_child() const noexcept
	{
		return m_num_elems_per_child;
	}
	/**
	 * Calculate the position of a child grid (i.e. partition) given the position of leaf grid node.
	 *
	 * @param pos_leaf_ leaf grid node position vector.
	 * @return position index of spatial partition in which leaf position lies.
	 */
	[[nodiscard]] PosIdx pos_idx_child(const VecDi & pos_leaf_) const noexcept
	{
		// Encode child position as an index.
		return felt2::index(pos_child(pos_leaf_), m_size_impl.size(), m_size_impl.offset());
	}

	/**
	 * Calculate the position of a child grid (i.e. partition) given the position of leaf grid node.
	 *
	 * @param pos_leaf_ leaf grid node position vector.
	 * @return position vector of spatial partition in which leaf position lies.
	 */
	[[nodiscard]] VecDi pos_child(const VecDi & pos_leaf_) const noexcept
	{
		// Position of leaf, without offset.
		auto pos_leaf_offset = pos_leaf_ - m_size_impl.offset();
		// Position of child grid containing leaf, without offset.
		auto pos_child_offset = (pos_leaf_offset.array() / m_child_size.array()).matrix();
		// Position of child grid containing leaf, including offset.
		auto pos_child = pos_child_offset + m_child_offset;
		// Encode child position as an index.
		return pos_child;
	}

	template <IsGridOfSpanGrids Children>
	Children make_children_span(
		HasResizeableStorage auto & storage_impl, auto &&... children_args) const
	{
		storage_impl.storage().resize(m_num_elems_per_child * m_num_children);
		std::span const parent_data{storage_impl.storage()};

		Children children{
			m_children_size,
			m_size_impl.offset(),
			std::forward<decltype(children_args)>(children_args)...};

		// Set each child sub-grid's size and offset.
		for (PosIdx pos_child_idx = 0; pos_child_idx < children.storage().size(); pos_child_idx++)
		{
			// Position of child in children grid.
			VecDi const pos_child = children.index(pos_child_idx);
			// Position of child in children grid, without offset.
			auto const pos_child_offset = pos_child - children.offset();
			// Scaled position of child == position in world space, without offset.
			auto const offset_child_offset =
				(pos_child_offset.array() * m_child_size.array()).matrix();
			// Position of child in world space, including offset.
			auto offset_child = offset_child_offset + m_size_impl.offset();

			// Calculate overflow at edge of grid.
			auto pos_lower = (pos_child.array() * m_child_size.array()).matrix();
			auto pos_upper = (pos_lower.array() + m_child_size.array()).matrix();
			auto signed_overflow = pos_upper - m_size_impl.size();
			auto overflow = signed_overflow.cwiseMax(0);

			auto & child = children.get(pos_child_idx);

			child.resize(m_child_size - overflow, offset_child);

			auto const num_used_child_idxs = static_cast<felt2::PosIdx>(child.size().prod());
			child.storage() =
				parent_data.subspan(pos_child_idx * m_num_elems_per_child, num_used_child_idxs);
		}

		return children;
	}
};

}  // namespace felt2::components