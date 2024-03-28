// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <etl/private/to_string_helper.h>
#include <etl/string.h>
#include <experimental/mdspan>
#include <sycl/sycl.hpp>

#include "./core.hpp"

namespace stdx = std::experimental;

namespace etl::private_to_string
{

template <typename TIString, typename T, felt2::Dim D>
const TIString & to_string(
	const felt2::VecDT<T, D> & value_,
	TIString & str_,
	const etl::basic_format_spec<TIString> & format_,
	bool append_ = false)
{
	str_ += "(";
	to_string(value_(0), str_, append_);
	for (decltype(value_.size()) dim = 1; dim < value_.size(); ++dim)
	{
		str_ += ", ";
		to_string(value_(dim), str_, format_, true);
	}
	str_ += ")";

	return str_;
}
}  // namespace etl::private_to_string

// NOTE: must come _after_ custom formatters.
#include <etl/to_string.h>

namespace felt2::device
{

template <typename T>
auto make_unique_sycl(
	std::size_t count_, sycl::device const & dev_, sycl::context const & ctx_, auto &&... args_)
{
	auto * mem_region = sycl::malloc_shared<T>(count_, dev_, ctx_);
	if (!mem_region)
		throw std::runtime_error{"make_unique_sycl: sycl::malloc_shared failed"};
	auto deleter = [ctx_](T * ptr_) { sycl::free(ptr_, ctx_); };
	auto ptr = std::unique_ptr<T, decltype(deleter)>{mem_region, std::move(deleter)};

	new (mem_region) T{std::forward<decltype(args_)>(args_)...};
	return ptr;
}

template <typename T>
auto make_unique_sycl(sycl::device const & dev_, sycl::context const & ctx_, auto &&... args_)
{
	return make_unique_sycl<T>(1, dev_, ctx_, std::forward<decltype(args_)>(args_)...);
}

template <typename T>
auto make_unique_sycl_array(
	sycl::device const & dev_, sycl::context const & ctx_, std::size_t const size_)
{
	auto deleter = [ctx_](T * ptr_) { sycl::free(ptr_, ctx_); };
	auto * mem_region = sycl::malloc_shared<T>(size_, dev_, ctx_);
	if (!mem_region)
		throw std::runtime_error{"make_unique_sycl_array: sycl::malloc_shared failed"};
	auto ptr =
		// TODO(DF): Could perhaps use std::array for this instead:
		// NOLINTNEXTLINE(*-avoid-c-arrays)
		std::unique_ptr<T[], decltype(deleter)>{mem_region, std::move(deleter)};
	return std::move(ptr);
}

template <typename T>
auto make_unique_sycl_array(sycl::queue const & queue_, std::size_t const size_)
{
	return make_unique_sycl_array<T>(queue_.get_device(), queue_.get_context(), size_);
}

template <typename T>
using UniqueSyclArrayT =
	decltype(make_unique_sycl_array<T>(std::declval<sycl::queue>(), std::declval<std::size_t>()));

}  // namespace felt2::device

namespace felt2::components::device
{
using felt2::device::make_unique_sycl;
using felt2::device::make_unique_sycl_array;
using felt2::device::UniqueSyclArrayT;

template <HasLog Logger, HasAbort Aborter>
struct Context
{
	Logger m_logger_impl;
	Aborter m_aborter_impl;
	sycl::device m_dev;
	sycl::context m_ctx;

	auto & logger(this auto & self_)
	{
		return self_.m_logger_impl;
	}
	auto & aborter(this auto & self_)
	{
		return self_.m_aborter_impl;
	}
	auto & device(this auto & self_)
	{
		return self_.m_dev;
	}
	auto & context(this auto & self_)
	{
		return self_.m_ctx;
	}
};

template <HasLeafType Traits>
struct USMRawArray
{
	using Leaf = typename Traits::Leaf;
	using Array = std::span<Leaf>;
	using UniquePtr = decltype(make_unique_sycl<Leaf>(
		std::declval<std::size_t>(), std::declval<sycl::device>(), std::declval<sycl::context>()));

	USMRawArray(std::size_t count_, sycl::device const & dev_, sycl::context const & ctx_)
		: m_ptr{make_unique_sycl<Leaf>(count_, dev_, ctx_)}, m_data{m_ptr.get(), count_}
	{
	}

	Array & storage()
	{
		return m_data;
	}

	[[nodiscard]] Array const & storage() const
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

	[[nodiscard]] Array const & storage() const
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

	void set_stream(sycl::stream * stream_)
	{
		m_stream = stream_;
	}

	sycl::stream * m_stream{nullptr};
};

struct Log
{
	struct Storage
	{
		UniqueSyclArrayT<char> char_data;
		// Cannot be std::vector since no copy constructor.
		UniqueSyclArrayT<etl::string_ext> str_data;
		std::span<etl::string_ext> strs;
	};

	std::span<etl::string_ext> strs;
	mutable sycl::private_ptr<std::size_t const> stream_id{nullptr};

	constexpr bool log(auto... args_) const noexcept
	{
		if (strs.empty())
			return false;

		std::size_t const stream_idx = stream_id ? *stream_id : 0;

		etl::string_ext & str = strs[stream_idx % strs.size()];

		(
			[&]
			{
				if constexpr (requires { std::string_view{args_}; })
				{
					std::string_view const arg_str{args_};
					str.append(arg_str.data(), arg_str.size());
				}
				else
				{
					etl::to_string(args_, str, true);
				}
			}(),
			...);

		return !str.full();
	}

	[[nodiscard]] constexpr bool has_logs() const noexcept
	{
		return !std::ranges::all_of(strs, [](auto const & str_) { return str_.empty(); });
	}

	[[nodiscard]] constexpr std::string_view text(std::size_t const stream_idx_) const noexcept
	{
		return std::string_view{strs[stream_idx_].data(), strs[stream_idx_].size()};
	}

	void set_storage(Storage const & storage_)
	{
		strs = storage_.strs;
	}

	void set_stream(std::size_t const * id_) const
	{
		stream_id = id_;
	}

	[[nodiscard]] static Storage make_storage(
		sycl::device const & dev_,
		sycl::context const & ctx_,
		std::size_t const num_streams_,
		std::size_t const max_msg_size_)
	{
		auto char_data = make_unique_sycl_array<char>(dev_, ctx_, num_streams_ * max_msg_size_);
		auto str_data = make_unique_sycl_array<etl::string_ext>(dev_, ctx_, num_streams_);
		std::span const str_data_span{str_data.get(), num_streams_};
		Storage storage{std::move(char_data), std::move(str_data), str_data_span};

		stdx::mdspan const char_data_span{storage.char_data.get(), num_streams_, max_msg_size_};

		for (std::size_t stream_idx = 0; stream_idx < num_streams_; ++stream_idx)
			new (&str_data_span[stream_idx])
				etl::string_ext{&char_data_span[stream_idx, 0], max_msg_size_};

		return storage;
	}
};

struct Aborter
{
	static void abort() noexcept
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