// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <cstddef>
#include <format>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "../../../sycl_util.hpp"

#include <convfelt/felt2/components/sycl.hpp>

SCENARIO("Assertion and logging in SYCL")
{
	GIVEN("queue and work range")
	{
		sycl::queue queue{sycl::gpu_selector_v, &async_handler};  // NOLINT(misc-const-correctness)

		static constexpr std::size_t k_num_work_items = 5;
		static constexpr std::size_t k_max_msg_size = 30;
		static constexpr std::size_t k_log_storage_size = 1024UL;

		sycl::range<1> work_items{k_num_work_items};

		using Allocator = sycl::usm_allocator<char, sycl::usm::alloc::shared>;
		std::vector<char, Allocator> const text_data(
			k_num_work_items * k_max_msg_size, '\0', Allocator{queue});

		WHEN("a kernel with a logger is executed")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), k_log_storage_size);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](const sycl::id<1> tid_)
						{
							auto const stream_id = static_cast<std::size_t>(tid_);
							logger.set_stream(&stream_id);
							logger.log("Hello from thread ", stream_id);
						});
				});
			queue.wait_and_throw();
			THEN("log is output")
			{
				for (std::size_t tid = 0; tid < work_items.get(0); ++tid)
				{
					CHECK(logger.text(tid) == fmt::format("Hello from thread {}", tid));
				}
			}
		}
		WHEN("a kernel logs to unexpected stream ids")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), k_log_storage_size);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{
							auto const stream_id = static_cast<std::size_t>(2 * tid_);
							logger.set_stream(&stream_id);
							logger.log("Hello from thread ", static_cast<int>(tid_), "\n");
						});
				});
			queue.wait_and_throw();

			THEN("log output stream is a circular buffer")
			{
				for (std::size_t tid = 0; tid < work_items.get(0); ++tid)
				{
					CAPTURE(tid);
					CHECK(
						logger.text((2 * tid) % work_items.get(0)) ==
						fmt::format("Hello from thread {}\n", tid));
				}
			}
		}

		WHEN("a kernel logs to thread-local stream and no stream is set")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), k_log_storage_size);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{ logger.log("Hello from thread ", static_cast<int>(tid_), "\n"); });
				});
			queue.wait_and_throw();

			THEN("all logs go to stream zero")
			{
				CAPTURE(logger.text(0));
				CHECK_THAT(
					std::string{logger.text(0)},
					Catch::Matchers::StartsWith(fmt::format("Hello from thread")));

				for (std::size_t tid = 1; tid < work_items.get(0); ++tid)
					CHECK(logger.text(tid).empty());
			}
		}

		WHEN("a kernel logs to thread-local stream then again with no stream set")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), k_log_storage_size);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{
							std::size_t const stream_id = tid_.get(0);
							logger.set_stream(&stream_id);

							logger.log("Hello from thread ", static_cast<int>(tid_), "\n");
						});
				});
			queue.wait_and_throw();
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{ logger.log("Hello again from thread ", static_cast<int>(tid_), "\n"); });
				});
			queue.wait_and_throw();

			THEN("second set of logs go to stream zero")
			{
				CAPTURE(logger.text(0));
				CHECK_THAT(
					std::string{logger.text(0)},
					Catch::Matchers::StartsWith(
						fmt::format("Hello from thread 0\nHello again from thread")));

				for (std::size_t tid = 1; tid < work_items.get(0); ++tid)
					CHECK(
						logger.text(tid) ==
						std::format("Hello from thread {}\n", static_cast<int>(tid)));
			}
		}
	}
}