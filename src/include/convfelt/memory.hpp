// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <memory>

#include <sycl/sycl.hpp>

namespace convfelt
{
template <typename T>
auto make_unique_sycl(sycl::device const & dev, sycl::context const & ctx, auto &&... args)
{
	auto * mem_region = sycl::malloc_shared<T>(1, dev, ctx);
	if (!mem_region) {
		throw std::runtime_error{"make_unique_sycl: sycl::malloc_shared failed"};
	}
	auto const deleter = [ctx](T * ptr) { sycl::free(ptr, ctx); };
	auto ptr = std::unique_ptr<T, decltype(deleter)>{mem_region, deleter};

	new (mem_region) T{std::forward<decltype(args)>(args)...};
	return ptr;
}
}  // namespace convfelt
