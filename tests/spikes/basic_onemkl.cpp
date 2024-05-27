// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "../sycl_util.hpp"

SCENARIO("Basic oneMKL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.F, 2.F, 3.F, 4.F, 5.F};
		std::vector<float> b = {-1.F, 2.F, -3.F, 4.F, -5.F};
		assert(a.size() == b.size());

		WHEN("vectors are added using oneMKL")
		{
			{
				sycl::queue queue{sycl::gpu_selector_v, &async_handler};  // NOLINT(misc-const-correctness)
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());

				// NOTE: if a segfault happens here it's because the ERROR_MSG is nullptr, which
				// means there are no enabled backend libraries.
				oneapi::mkl::blas::column_major::axpy(
					queue, static_cast<std::int64_t>(a.size()), 1.0F, buff_a, 1, buff_b, 1);
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {0.F, 4.F, 0.F, 8.F, 0.F};

				CHECK(b == expected);
			}
		}
	}
}
