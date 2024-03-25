// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <hipSYCL/sycl/libkernel/multi_ptr.hpp>
#include <span>

#include <etl/private/to_string_helper.h>
#include <etl/string.h>

#include <experimental/mdspan>
#include <string_view>
#include <sycl/sycl.hpp>

#include "./core.hpp"

namespace stdx = std::experimental;

namespace etl::private_to_string
{

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

namespace felt2::device
{

template <typename T>
auto make_unique_sycl(
	std::size_t count, sycl::device const & dev, sycl::context const & ctx, auto &&... args)
{
	auto * mem_region = sycl::malloc_shared<T>(count, dev, ctx);
	if (!mem_region)
	{
		throw std::runtime_error{"make_unique_sycl: sycl::malloc_shared failed"};
	}
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
	sycl::context ctx;
	void operator()(T * ptr)
	{
		sycl::free(ptr, ctx);
	}
};

template <typename T>
auto make_unique_sycl_array(sycl::queue const & queue, std::size_t const size)
{
	auto * mem_region = sycl::malloc_shared<T>(size, queue);
	auto ptr =
		std::unique_ptr<T[], SyclDeleter<T>>{mem_region, SyclDeleter<T>{queue.get_context()}};
	return std::move(ptr);
}

template <typename T>
auto make_unique_sycl_array(
	sycl::device const & dev, sycl::context const & ctx, std::size_t const size)
{
	auto * mem_region = sycl::malloc_shared<T>(size, dev, ctx);
	auto ptr = std::unique_ptr<T[], SyclDeleter<T>>{mem_region, SyclDeleter<T>{ctx}};
	return std::move(ptr);
}

template <typename T>
using unique_sycl_array_t =
	decltype(make_unique_sycl_array<T>(std::declval<sycl::queue>(), std::declval<std::size_t>()));

}  // namespace felt2::device

namespace felt2::components::device
{
using felt2::device::make_unique_sycl;
using felt2::device::make_unique_sycl_array;
using felt2::device::unique_sycl_array_t;

template <HasLog Logger, HasAbort Aborter>
struct Context
{
	Logger m_logger_impl;
	Aborter m_aborter_impl;
	sycl::device m_dev;
	sycl::context m_ctx;

	auto& logger(this auto & self) {
		return self.m_logger_impl;
	}
	auto& aborter(this auto & self) {
		return self.m_aborter_impl;
	}
	auto& device(this auto & self) {
		return self.m_dev;
	}
	auto& context(this auto & self) {
		return self.m_ctx;
	}
};


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
		unique_sycl_array_t<char> char_data;
		// Cannot be std::vector since no copy constructor.
		unique_sycl_array_t<etl::string_ext> str_data;
		std::span<etl::string_ext> strs;
	};

	std::span<etl::string_ext> strs;
	mutable sycl::private_ptr<std::size_t const> stream_id;

	template <class... Args>
	constexpr bool log(Args &&... args) const noexcept
	{
		if (strs.empty())
			return false;

		std::size_t const stream_idx = stream_id ? *stream_id : 0;

		etl::string_ext & str = strs[stream_idx % strs.size()];

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
		return !std::ranges::all_of(strs, [](auto const & str) { return str.empty(); });
	}

	[[nodiscard]] constexpr std::string_view text(std::size_t const stream_idx) const noexcept
	{
		return std::string_view{strs[stream_idx].data(), strs[stream_idx].size()};
	}

	void set_storage(Storage const & storage)
	{
		strs = storage.strs;
	}

	void set_stream(std::size_t const * id) const
	{
		stream_id = id;
	}

	[[nodiscard]] static Storage make_storage(
		sycl::device const & dev,
		sycl::context const & ctx,
		std::size_t const num_streams,
		std::size_t const max_msg_size)
	{
		auto char_data = make_unique_sycl_array<char>(dev, ctx, num_streams * max_msg_size);
		auto str_data = make_unique_sycl_array<etl::string_ext>(dev, ctx, num_streams);
		std::span const str_data_span{str_data.get(), num_streams};
		Storage storage{std::move(char_data), std::move(str_data), str_data_span};

		stdx::mdspan const char_data_span{storage.char_data.get(), num_streams, max_msg_size};

		for (std::size_t stream_idx = 0; stream_idx < num_streams; ++stream_idx)
			new (&str_data_span[stream_idx])
				etl::string_ext{&char_data_span[stream_idx, 0], max_msg_size};

		return storage;
	}
};

struct Aborter
{
	void abort() const noexcept
	{
#ifndef FELT2_DEBUG_NONFATAL
		// TODO(DF): Need a less vendor-specific solution. AdaptiveCpp generic JIT backend is
		//  __SYCL_SINGLE_SOURCE__ yet provides (as below) vendor-soecific macros to target code to
		//  host vs. device.
		__hipsycl_if_target_device(asm("trap;"));
		__hipsycl_if_target_host(std::abort());
#endif
	}
};

}  // namespace felt2::components::device