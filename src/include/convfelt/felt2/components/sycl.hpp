// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <ostream>

#include <sycl/sycl.hpp>

#include "./core.hpp"
namespace felt2
{

template <typename T>
auto make_unique_sycl(
	std::size_t count, sycl::device const & dev, sycl::context const & ctx, auto &&... args)
{
	auto * mem_region = sycl::malloc_shared<T>(count, dev, ctx);
	auto deleter = [ctx](T * ptr) { sycl::free(ptr, ctx); };
	auto ptr = std::unique_ptr<T, decltype(deleter)>{mem_region, std::move(deleter)};

	new (mem_region) T{std::forward<decltype(args)>(args)...};
	return ptr;
}

template <typename T>
auto make_unique_sycl(sycl::device const & dev, sycl::context const & ctx, auto &&... args)
{
	return make_unique_sycl<T>(1, dev, ctx, std::forward<decltype(args)>(args)...);
}
}  // namespace felt2

namespace felt2::components
{

template <HasLeafType Traits>
struct USMRawArray
{
	using Leaf = typename Traits::Leaf;
	using Array = std::span<Leaf>;
	using UniquePtr = decltype(make_unique_sycl<Leaf>(
		std::declval<std::size_t>(), std::declval<sycl::device>(), std::declval<sycl::context>()));

	USMRawArray(std::size_t count, sycl::device const & dev, sycl::context const & ctx)
		: m_ptr{make_unique_sycl<Leaf>(count, dev, ctx)}, m_data{m_ptr.get(), count}
	{
	}

	Array & storage()
	{
		return m_data;
	}

	Array const & storage() const
	{
		return m_data;
	}

private:
	UniquePtr m_ptr;
	Array m_data;
};

template <HasLeafType Traits>
struct USMResizeableArray
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