// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <span>
#include <type_traits>
#include <vector>

#include "../index.hpp"
#include "../typedefs.hpp"

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
concept IsData = requires(T obj) {
					 {
						 obj.data()
					 } -> IsPointer;
					 {
						 obj.size()
					 } -> std::convertible_to<PosIdx>;
				 };

template <class T>
concept IsResizeable = requires(T obj) {
						   {
							   obj.resize(std::declval<PosIdx>())
						   };
					   };

template <class T>
concept HasDataType = IsData<typename T::DataArray>;

template <class T>
concept HasData = requires(T obj) {
					  {
						  obj.data()
					  } -> IsData;
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
						  t.inside(std::declval<typename T::VecDi>())
					  } -> std::convertible_to<bool>;
					  {
						  t.index(std::declval<typename T::VecDi>())
					  } -> std::convertible_to<PosIdx>;
					  {
						  t.index(std::declval<PosIdx>())
					  } -> std::convertible_to<typename T::VecDi>;
				  };

template <class T, Dim D>
concept HasAssertBounds =
	requires(T t) {
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

	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(m_size, m_offset, m_offset_plus_size);
	}

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
		return felt2::index<k_dims>(pos_, size(), offset());
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
		return felt2::index<k_dims>(idx_, size(), offset());
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

	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(m_size, m_offset, m_offset_plus_size);
	}

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
		return felt2::index<k_dims>(pos_, size(), offset());
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
		return felt2::index<k_dims>(idx_, size(), offset());
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

template <HasDims Traits, HasStream Stream, HasSize Size, HasData Data>
struct AssertBounds
{
	static constexpr Dim k_dims = Traits::k_dims;
	using VecDi = VecDi<k_dims>;

	Stream & m_stream_impl;
	Size const & m_size_impl;
	Data const & m_data_impl;

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
		if (m_stream_impl.has_stream() && !m_size_impl.inside(pos_))
		{
			m_stream_impl.get_stream()
				<< "AssertionError: " << title_ << " assert_pos_bounds(" << pos_(0);
			for (Dim axis = 1; axis < pos_.size(); ++axis)
				m_stream_impl.get_stream() << ", " << pos_(axis);
			m_stream_impl.get_stream() << ")\n";
		}
		assert(m_size_impl.inside(pos_));
	}

	void assert_pos_idx_bounds(const PosIdx pos_idx_, const char * title_) const
	{
		if (m_stream_impl.has_stream() && pos_idx_ >= m_data_impl.data().size())
		{
			auto pos = m_size_impl.index(pos_idx_);
			m_stream_impl.get_stream() << "AssertionError: " << title_ << " assert_pos_idx_bounds("
									   << pos_idx_ << ") i.e. ";
			format_pos(m_stream_impl.get_stream(), pos);
			m_stream_impl.get_stream() << "\n";
		}
		assert(pos_idx_ < m_data_impl.data().size());
	}
};

template <HasDimsAndLeafType Traits, HasSize Size, HasData Data, HasStream Stream>
struct Activate
{
	/// Dimension of the grid.
	static const Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;

	/// D-dimensional signed integer vector.
	using VecDi = VecDi<k_dims>;

	Size const & m_size_impl;
	Data & m_data_impl;
	Stream & m_stream_impl;
	Leaf const m_background;

	/**
	 * Serialisation hook for cereal library.
	 *
	 * @param ar
	 */
	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(m_background);
	}

	/**
	 * Get whether this grid has been activated (data allocated) or not.
	 *
	 * @return true if data allocated, false if not.
	 */
	[[nodiscard]] bool is_active() const noexcept
	{
		return bool(m_data_impl.data().size());
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
		auto const & size = m_size_impl.size();
		NodeIdx arr_size = size(0);
		for (Dim i = 1; i < size.size(); i++) arr_size *= size(i);
		m_data_impl.data().resize(PosIdx(arr_size), m_background);
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
	HasData Data,
	HasAssertBounds<Traits::k_dims> Assert>
struct AccessByRef
{
	/// Dimension of the grid.
	static const Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// D-dimensional signed integer vector.
	using VecDi = Felt::VecDi<k_dims>;

	Size const & m_size_impl;
	Data & m_data_impl;
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
		return m_data_impl.data()[pos_idx_];
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
		return m_data_impl.data()[pos_idx_];
	}
};

template <
	HasDimsAndLeafType Traits,
	HasSize Size,
	HasData Data,
	HasAssertBounds<Traits::k_dims> Assert>
struct AccessByValue
{
	/// Dimension of the grid.
	static const Dim k_dims = Traits::k_dims;
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// D-dimensional signed integer vector.
	using VecDi = Felt::VecDi<k_dims>;

	Size const & m_size_impl;
	Data & m_data_impl;
	Assert const & m_assert_impl;

	/**
	 * Get the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @return internally stored value at given grid position
	 */
	Leaf get(const VecDi & pos_) const noexcept
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_bounds(pos_, "get: ");
#endif
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
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_idx_bounds(pos_idx_, "get: ");
#endif
		return m_data_impl.data()[pos_idx_];
	}

	/**
	 * Set the value stored in the grid.
	 *
	 * @param pos_ position in grid to query.
	 * @param val_ value to copy into grid at pos_.
	 */
	void set(const VecDi & pos_, Leaf val_)
	{
#ifdef FELT2_DEBUG_ENABLED
		m_assert_impl.assert_pos_bounds(pos_, "set: ");
#endif
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
		m_data_impl.data()[pos_idx_] = val_;
	}
};

template <HasLeafType Traits>
struct DataArray
{
	using Leaf = typename Traits::Leaf;
	using Array = std::vector<Leaf>;

	Array m_data{};

	Array & data()
	{
		return m_data;
	}

	const Array & data() const
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

template <HasLeafType Traits>
struct DataArraySpan
{
	using Leaf = typename Traits::Leaf;
	using Array = std::span<Leaf>;

	Array m_data{};

	Array & data()
	{
		return m_data;
	}

	const Array & data() const
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

}  // namespace felt2::components