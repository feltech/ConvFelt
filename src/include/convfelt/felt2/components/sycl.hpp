// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <ostream>

#include <sycl/sycl.hpp>

#include "./core.hpp"

namespace felt2::components
{

template <HasLeafType Traits>
struct USMDataArray
{
	using Leaf = typename Traits::Leaf;
	using Allocator = sycl::usm_allocator<Leaf, sycl::usm::alloc::shared>;
	using Array = std::vector<Leaf, Allocator>;

	Allocator m_allocator;
	Array m_data{m_allocator};

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

struct Stream
{
#ifdef SYCL_DEVICE_ONLY
	using StreamType = sycl::stream;
#else
	using StreamType = std::ostream;
#endif

	[[nodiscard]] bool has_stream() const
	{
		return m_stream != nullptr;
	}

	[[nodiscard]] StreamType & get_stream() const
	{
		return *m_stream;
	}

	void set_stream(StreamType * stream)
	{
		m_stream = stream;
	}

#ifdef SYCL_DEVICE_ONLY
	StreamType * m_stream{nullptr};
#else
	StreamType * m_stream{&std::cerr};
#endif
};
}  // namespace felt2::components