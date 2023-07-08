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

	Array & storage()
	{
		return m_data;
	}

	Array const & storage() const
	{
		return m_data;
	}
};

struct Stream
{
	[[nodiscard]] bool has_stream() const
	{
#ifdef SYCL_DEVICE_ONLY
		return m_stream != nullptr;
#else
		return true;
#endif
	}

#ifdef SYCL_DEVICE_ONLY
	[[nodiscard]] sycl::stream & get_stream() const
	{
		assert(has_stream());
		return *m_stream;
	}
#else
	[[nodiscard]] std::ostream & get_stream() const
	{
		return std::cerr;
	}
#endif

	void set_stream(sycl::stream * stream)
	{
		m_stream = stream;
	}

	sycl::stream * m_stream{nullptr};
};
}  // namespace felt2::components