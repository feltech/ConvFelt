#include <type_traits>
#include <ranges>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>

#include <Eigen/Eigen>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/iter.hpp>

#include "../sycl_util.hpp"

SCENARIO("Accessing underlying grid data as a (num children)x(child size) matrix")
{
	GIVEN("a ConvGrid of size (8,8,4) with 2x2 filter partitions")
	{
		using PowTwoSize = convfelt::ConvGrid::PowTwoDu;
		using PowTwoWindowSize = convfelt::ConvGrid::PowTwoWindowSize;
		convfelt::ConvGrid grid{
			convfelt::make_host_context(),
			PowTwoSize::from_exponents({3, 3, 2}),
			PowTwoWindowSize::from_exponents({1, 1})};

		WHEN("data is queried")
		{
			THEN("it is a column-major FxP matrix")
			{
				auto const & data = grid.matrix();
				STATIC_REQUIRE(!std::decay_t<decltype(data)>::IsRowMajor);
				CHECK(data.rows() == grid.child_size().as_pos().prod());
				CHECK(data.cols() == grid.children().size().as_pos().prod());
			}
		}

		WHEN("data is modified within a filter partition")
		{
			grid.children().get({0, 2, 0}).set(1U, 123.4F);

			THEN("expected column of parent grid's data is updated")
			{
				CHECK(grid.matrix()(1, 2) == 123.4F);
			}
		}
	}
}

SCENARIO("Using SYCL to update values in a grid")
{
	GIVEN("Host/device shared grid filled with '3'")
	{
		sycl::context const ctx;
		sycl::device const dev{sycl::gpu_selector_v};
		using ConvGrid = convfelt::ConvGridTD<float, 3, convfelt::GridFlag::is_device_shared>;

		auto pgrid = felt2::device::make_unique_sycl<ConvGrid>(
			dev,
			ctx,
			convfelt::make_device_context(dev, ctx),
			ConvGrid::PowTwoDu::from_exponents({2, 2, 2}),
			ConvGrid::PowTwoWindowSize::from_exponents({1, 1}));

		std::ranges::fill(pgrid->storage(), 3);

		THEN("grid storage has expected size")
		{
			REQUIRE(!pgrid->children().storage().empty());
			CHECK(pgrid->children().storage().size() == 4);
			CHECK(pgrid->children().storage()[0].storage().size() > 1);
			CHECK(pgrid->children().storage()[0].storage().data() == pgrid->storage().data());
		}

		WHEN("grid data is doubled using sycl")
		{
			sycl::range<1> const work_items{pgrid->children().storage().size()};

			sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

			[[maybe_unused]] auto log_storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			pgrid->context().logger().set_storage(log_storage);

			queue.submit([&](sycl::handler & cgh_)
						 { cgh_.prefetch(pgrid->storage().data(), pgrid->storage().size()); });
			queue.submit(
				[&](sycl::handler & cgh_) {
					cgh_.prefetch(
						pgrid->children().storage().data(), pgrid->children().storage().size());
				});
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class grid_mult>(
						work_items,
						[pgrid = pgrid.get()](sycl::id<1> tid_)
						{
							for (auto & val : convfelt::iter::val(pgrid->children().get(tid_)))
								val *= 2;
						});
				});

			queue.wait_and_throw();

			THEN("all elements of the grid contain '6'")
			{
				CHECK(!pgrid->context().logger().has_logs());

				for (auto const val : convfelt::iter::val(*pgrid)) CHECK(val == 6);
			}
		}

		WHEN("out of bounds access in kernel")
		{
			//			sycl::device cpu_dev{sycl::cpu_selector_v};
			sycl::range<1> const work_items{pgrid->children().storage().size()};
			sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

			[[maybe_unused]] auto log_storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			pgrid->context().logger().set_storage(log_storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class grid_mult>(
						work_items,
						[pgrid = pgrid.get()](sycl::id<1> tid_)
						{
							auto const log_id = static_cast<std::size_t>(tid_);
							pgrid->context().logger().set_stream(&log_id);

							auto child_idx = static_cast<felt2::PosIdx>(tid_) + 1;
							pgrid->children().get(child_idx);
						});
				});

			// Note: FELT2_DEBUG_NONFATAL macro enabled so assertion doesn't cause a kernel crash,
			// since kernel crashes are "sticky" in CUDA and would require the whole host process to
			// be restarted.
			queue.wait_and_throw();

			THEN("out of bounds access is logged")
			{
				CHECK(pgrid->context().logger().text(0U).empty());
				CHECK(pgrid->context().logger().text(1U).empty());
				CHECK(pgrid->context().logger().text(2U).empty());
				CHECK(
					pgrid->context().logger().text(3U) ==
					// Note: the "i.e. (0, 0, 0)" is due to modulo arithmetic
					"AssertionError: get:  assert_pos_idx_bounds(4) i.e. (0, 0, 0) is greater than "
					"extent 4\n");
			}
		}
	}
}