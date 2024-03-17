// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <ostream>
#include <ranges>

#include <etl/private/to_string_helper.h>
#include <etl/string.h>

#include <experimental/mdspan>
#include <string_view>
#include <sycl/sycl.hpp>

#include "./core.hpp"

namespace stdx = std::experimental;

namespace etl::private_to_string
{
// template <typename T, std::size_t D>
// const etl::istring & to_string(
//	const felt2::VecDT<T, D> & value, etl::istring & str, bool append = false)
//{
//	etl::format_spec format;
//	return to_string(value, str, format, append);
// }

template <typename TIString, typename T, felt2::Dim D>
const TIString & to_string(
	const felt2::VecDT<T, D> & value,
	TIString & str,
	const etl::basic_format_spec<TIString> & format,
	bool append = false)
{
	str += "(";
	to_string(value(0), str, append);
	for (decltype(value.size()) dim = 1; dim < value.size(); ++dim)
	{
		str += ", ";
		to_string(value(dim), str, format, true);
	}
	str += ")";

	return str;
}
}  // namespace etl::private_to_string

// NOTE: must come _after_ custom formatters.
#include <etl/to_string.h>

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

template <class T>
struct SyclDeleter
{
	sycl::queue queue;
	void operator()(T * ptr)
	{
		sycl::free(ptr, queue);
	}
};

template <typename T>
auto make_unique_sycl_array(sycl::queue const & queue, std::size_t const size)
{
	auto * mem_region = sycl::malloc_shared<T>(size, queue);
	auto ptr = std::unique_ptr<T[], SyclDeleter<T>>{mem_region, SyclDeleter<T>{queue}};
	return std::move(ptr);
}

template <typename T>
using unique_sycl_array_t =
	decltype(make_unique_sycl_array<T>(std::declval<sycl::queue>(), std::declval<std::size_t>()));

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
		return m_stream != nullptr;
	}

	[[nodiscard]] sycl::stream & get_stream() const
	{
		assert(has_stream());
		return *m_stream;
	}

	void set_stream(sycl::stream * stream)
	{
		m_stream = stream;
	}

	sycl::stream * m_stream{nullptr};
};

struct Log
{
	struct Storage
	{
		unique_sycl_array_t<char> data;
		unique_sycl_array_t<etl::string_ext> strs;
	};
	std::span<etl::string_ext> str_span_;

	template <class... Args>
	constexpr bool log(std::size_t const stream_idx, Args... args) const noexcept
	{
		etl::string_ext & str = str_span_[stream_idx];
		(
			[&]
			{
				if constexpr (requires { std::string_view{args}; })
				{
					std::string_view const arg_str{args};
					str.append(arg_str.data(), arg_str.size());
				}
				else
				{
					etl::to_string(args, str, true);
				}
			}(),
			...);

		return !str.full();
	}

	[[nodiscard]] constexpr bool has_logs() const noexcept
	{
		return !std::ranges::all_of(str_span_, [](auto const & str) { return str.empty(); });
	}

	[[nodiscard]] constexpr std::string_view text(std::size_t const stream_idx) const
	{
		return std::string_view{str_span_[stream_idx].data(), str_span_[stream_idx].size()};
	}

	[[nodiscard]] Storage reset(
		sycl::queue const & queue, std::size_t const num_streams, std::size_t const max_msg_size)
	{
		Storage storage{
			make_unique_sycl_array<char>(queue, num_streams * max_msg_size),
			make_unique_sycl_array<etl::string_ext>(queue, num_streams)};

		stdx::mdspan const data_span{storage.data.get(), num_streams, max_msg_size};
		str_span_ = std::span{storage.strs.get(), num_streams};

		for (std::size_t stream_idx = 0; stream_idx < num_streams; ++stream_idx)
			new (&str_span_[stream_idx]) etl::string_ext{&data_span(stream_idx, 0), max_msg_size};

		return storage;
	}
};
}  // namespace felt2::components