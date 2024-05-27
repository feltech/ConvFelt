// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sycl/sycl.hpp>

#include <convfelt/felt2/components/sycl.hpp>

#include "../sycl_util.hpp"


SCENARIO("Basic SyCL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.F, 2.F, 3.F, 4.F, 5.F};
		std::vector<float> b = {-1.F, 2.F, -3.F, 4.F, -5.F};
		std::vector<float> c(a.size());
		assert(a.size() == b.size());
		sycl::queue queue{sycl::gpu_selector_v, &async_handler};  // NOLINT(misc-const-correctness)

		WHEN("vectors are added using sycl")
		{
			using Allocator = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
			std::vector<float, Allocator> vals(Allocator{queue});
			{
				sycl::range<1> work_items{a.size()};
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());
				sycl::buffer<float> buff_c(c.data(), c.size());

				vals.push_back(1);
				vals.push_back(2);

				queue.submit(
					[&](sycl::handler & cgh_)
					{
						auto access_a = buff_a.get_access<sycl::access::mode::read>(cgh_);
						auto access_b = buff_b.get_access<sycl::access::mode::read>(cgh_);
						auto access_c = buff_c.get_access<sycl::access::mode::write>(cgh_);

						cgh_.parallel_for<class vector_add>(
							work_items,
							[=, data = vals.data()](sycl::id<1> tid_)
							{
								access_c[tid_] =
									// NOLINTNEXTLINE(*-pro-bounds-pointer-arithmetic)
									access_a[tid_] + access_b[tid_] + data[0] + data[1];
							});
					});
				queue.wait_and_throw();
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {3.F, 7.F, 3.F, 11.F, 3.F};

				CHECK(c == expected);
			}
		}

		WHEN("USM vector is doubled by kernel")
		{
			using Allocator = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
			auto vals = felt2::device::make_unique_sycl<std::vector<float, Allocator>>(
				queue.get_device(), queue.get_context(), Allocator{queue});
			vals->push_back(1);
			vals->push_back(2);
			sycl::range<1> work_items{vals->size()};
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_double>(
						work_items,
						[pvals = vals.get()](sycl::id<1> tid_)
						{
							auto & vals = *pvals;
							auto const val = vals[tid_];
							vals[tid_] = val * static_cast<float>(vals.size());
						});
				});
			queue.wait_and_throw();

			THEN("values have been doubled")
			{
				std::vector<float, Allocator> expected{Allocator{queue}};
				expected.push_back(2);
				expected.push_back(4);

				CHECK(*vals == expected);
			}
		}
		WHEN("CUDA error")
		{
			using Allocator = sycl::usm_allocator<char, sycl::usm::alloc::shared>;
			std::vector<char, Allocator> vals(Allocator{queue});
			static constexpr std::size_t k_buff_len = 20;
			vals.resize(k_buff_len);

			sycl::range<1> work_items{vals.size()};
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_double>(
						work_items,
						[buff = vals.data()](sycl::id<1> tid_) {
							std::format_to_n(
								buff, k_buff_len, "Hello from thread {}", static_cast<int>(tid_));
						});
				});

			THEN("Error is reported")
			{
				try
				{
					queue.wait_and_throw();
					FAIL("Should have thrown");
				}
				catch (std::exception const & exc)
				{
					std::string const error_message{exc.what()};
					CHECK_THAT(
						error_message,
						Catch::Matchers::ContainsSubstring(
							"Unresolved extern function 'memchr' (error code = CU:218)"));
				}
			}
		}
	}
}
