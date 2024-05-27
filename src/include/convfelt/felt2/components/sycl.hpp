// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include <etl/basic_format_spec.h>
#include <etl/string.h>
#include <experimental/mdspan>
#include <sycl/sycl.hpp>
// TODO(DF): Using std::views gives `error: type constraint differs in template redeclaration` - try
//  again with later version of libstdc++.
#include <range/v3/view/zip.hpp>

#include "./core.hpp"

namespace stdx = std::experimental;

namespace etl::private_to_string
{

/**
 * Transforms a vector into a string representation, with elements enclosed by parentheses and
 * separated by commas.
 *
 * This is a hook for the ETL string library to provide string conversions for vectors.
 *
 * @tparam TIString Container type for the converted string output.
 * @tparam T Element data type inside the input vector.
 * @tparam D Dimension of the input vector.
 * @param value_ Vector to stringifyt.
 * @param str_ Container for the converted string output.
 * @param format_ Format specifier in line with each element conversion in the object.
 * @param append_ Defines whether the converted string should be appended to the existing content of
 * str_. (Default: false)
 * @return Reference to the input @p str_, containing the converted string.
 */
template <typename TIString, typename T, felt2::Dim D>
TIString const & to_string(
	felt2::VecDT<T, D> const & value_,
	TIString & str_,
	etl::basic_format_spec<TIString> const & format_,
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
/**
 * Dynamically allocates and constructs an object on a SYCL device, with the object being created
 * using provided arguments.
 *
 * @tparam T Object being created.
 * @param dev_ SYCL device for memory allocation.
 * @param ctx_ SYCL context for memory allocation.
 * @param args_ Arguments for object construction.
 * @return A unique pointer to the created object.
 * @throws std::runtime_error If sycl::malloc_shared fails to allocate memory.
 */
template <typename T>
auto make_unique_sycl(sycl::device const & dev_, sycl::context const & ctx_, auto &&... args_)
{
	auto * mem_region = sycl::malloc_shared<T>(1, dev_, ctx_);
	if (!mem_region)
		throw std::runtime_error{"make_unique_sycl: sycl::malloc_shared failed"};
	auto deleter = [ctx_](T * ptr_) { sycl::free(ptr_, ctx_); };
	auto ptr = std::unique_ptr<T, decltype(deleter)>{mem_region, std::move(deleter)};

	new (mem_region) T{std::forward<decltype(args_)>(args_)...};
	return ptr;
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
	using UniquePtr = decltype(make_unique_sycl_array<Leaf>(
		std::declval<sycl::device>(), std::declval<sycl::context>(), std::declval<std::size_t>()));

	USMRawArray(std::size_t count_, sycl::device const & dev_, sycl::context const & ctx_)
		: m_ptr{make_unique_sycl_array<Leaf>(dev_, ctx_, count_)}, m_data{m_ptr.get(), count_}
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
		template <typename T>
		using UsmArray = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;

		UsmArray<char> buffer;
		// Use optional to work around lack of default/copy construction of string_ext when
		// creating a fixed size vector.
		UsmArray<std::optional<etl::string_ext>> strs;
	};

	std::span<std::optional<etl::string_ext>> strs;
	mutable sycl::private_ptr<std::size_t const> stream_id{nullptr};

	JB_HAS_SIDE_EFFECTS
	constexpr bool log(auto... args_) const noexcept
	{
		if (strs.empty())
			return false;

		std::size_t const stream_idx = stream_id ? *stream_id : 0;

		etl::string_ext & str = *strs[stream_idx % strs.size()];

		(
			[&](auto const & arg_)
			{
				if constexpr (requires { std::string_view{arg_}; })
				{
					std::string_view const arg_str{arg_};
					str.append(arg_str.data(), arg_str.size());
				}
				else
				{
					etl::to_string(arg_, str, true);
				}
			}(args_),
			...);

		return !str.full();
	}

	[[nodiscard]] constexpr bool has_logs() const noexcept
	{
		return !std::ranges::all_of(strs, [](auto const & str_) { return str_->empty(); });
	}

	[[nodiscard]] constexpr std::string_view text(std::size_t const stream_idx_) const noexcept
	{
		return std::string_view{strs[stream_idx_]->data(), strs[stream_idx_]->size()};
	}

	void set_storage(Storage & storage_)
	{
		strs = storage_.strs;
	}

	void set_stream(std::size_t const * id_) const
	{
		stream_id = id_;
	}

	std::size_t num_streams() const
	{
		return strs.size();
	}

	[[nodiscard]] static Storage make_storage(
		sycl::device const & dev_,
		sycl::context const & ctx_,
		std::size_t const num_streams_,
		std::size_t const max_msg_size_)
	{
		Storage storage{
			.buffer = decltype(Storage::buffer)(num_streams_ * max_msg_size_, '\0', {ctx_, dev_}),
			.strs = decltype(Storage::strs)(num_streams_, {ctx_, dev_})};

		// TODO(DF): Gone a bit overboard, could be simpler, but provides a reference for ranges and
		// mdspan.

		auto char_span_by_stream =
			std::views::iota(0U, storage.strs.size()) |
			std::views::transform(
				[chars_by_stream =
					 stdx::mdspan{storage.buffer.data(), num_streams_, max_msg_size_}](
					std::size_t idx)
				{ return stdx::submdspan(chars_by_stream, idx, stdx::full_extent); }) |
			std::views::transform(
				[](auto && stream_mdspan)
				{ return std::span(stream_mdspan.data_handle(), stream_mdspan.size()); });

		for (auto && [stream_str, stream_chars] :
			 ranges::views::zip(storage.strs, char_span_by_stream))
			stream_str.emplace(stream_chars.data(), stream_chars.size());

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
		__hipsycl_if_target_device(asm("trap;");) __hipsycl_if_target_host(std::abort();)
#endif
	}
};

}  // namespace felt2::components::device