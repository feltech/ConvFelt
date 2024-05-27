// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <convfelt/FilterSizeHelper.hpp>
#include <convfelt/felt2/typedefs.hpp>

SCENARIO("Transforming source image to/from filter input grid points")
{
	GIVEN("stride size 3x3, filter size 3x3 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{3, 3, 0};
		felt2::Vec3i const filter_size{3, 3, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("only one stamp is required")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input size is same as source size")
			{
				CHECK(input_size == felt2::Vec3i{3, 3, 4});
			}
		}
	}

	GIVEN("stride size 1x1, filter size 3x3 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{1, 1, 0};
		felt2::Vec3i const filter_size{3, 3, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("only one stamp is required")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input size is same as source size")
			{
				CHECK(input_size == felt2::Vec3i{3, 3, 4});
			}
		}
	}

	GIVEN("stride size 2x2, filter size 3x3 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{2, 2, 0};
		felt2::Vec3i const filter_size{3, 3, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("only one stamp is required")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input size is same as source size")
			{
				CHECK(input_size == felt2::Vec3i{3, 3, 4});
			}
		}
	}

	GIVEN("stride size 1x1, filter size 1x1 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{1, 1, 0};
		felt2::Vec3i const filter_size{1, 1, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("3 stamps in each direction is required")
			{
				CHECK(size == felt2::Vec3i{3, 3, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input grid size is same as source size")
			{
				// Total size is same as input, but every 1x1 window is a separate filter.
				CHECK(input_size == felt2::Vec3i{3, 3, 4});
			}
		}
	}

	GIVEN("stride size 2x2, filter size 2x2 and image size 4x4 with 4 channels")
	{
		felt2::Vec3i const filter_stride{2, 2, 0};
		felt2::Vec3i const filter_size{2, 2, 4};

		felt2::Vec3i const source_size{4, 4, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("2 stamps in each direction is required")
			{
				CHECK(size == felt2::Vec3i{2, 2, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input grid size is same as source size")
			{
				CHECK(input_size == felt2::Vec3i{4, 4, 4});
			}
		}
	}

	GIVEN("stride size 1x2, filter size 2x2 and image size 4x4 with 4 channels")
	{
		felt2::Vec3i const filter_stride{1, 2, 0};
		felt2::Vec3i const filter_size{2, 2, 4};

		felt2::Vec3i const source_size{4, 4, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("3 horizontal and 2 vertical stamps are required")
			{
				CHECK(size == felt2::Vec3i{3, 2, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input grid size is larger than source to accommodate overlapping filter stamps")
			{
				CHECK(input_size == felt2::Vec3i{6, 4, 4});
			}
		}
	}

	GIVEN("stride size 2x3, filter size 3x3 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{2, 3, 0};
		felt2::Vec3i const filter_size{3, 3, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("only one stamp is required")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input grid size is same as source size")
			{
				CHECK(input_size == felt2::Vec3i{3, 3, 4});
			}
		}
	}

	GIVEN("stride size 3x3, filter size 3x2 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{3, 3, 0};
		felt2::Vec3i const filter_size{3, 2, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filter stamps is calculated")
		{
			felt2::Vec3i const size =
				FilterSizeHelper<>::num_filter_regions_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("only one stamp is required")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				FilterSizeHelper<>::input_size_from_source_and_filter_size(
					filter_size, filter_stride, source_size);

			THEN("input grid size is too small due to stride step going out of bounds")
			{
				// Filter(s) doesn't cover input - should warn/abort.
				CHECK(input_size == felt2::Vec3i{3, 2, 4});
			}
		}
	}

	GIVEN("stride size 1x2, filter size 2x2 and image size 4x4 with 4 channels")
	{
		felt2::Vec3i const filter_stride{1, 2, 0};
		felt2::Vec3i const filter_size{2, 2, 4};

		felt2::Vec3i const source_size{4, 4, 4};

		felt2::Vec3i const input_size = FilterSizeHelper<>::input_size_from_source_and_filter_size(
			filter_size, filter_stride, source_size);

		CHECK(input_size == felt2::Vec3i{6, 4, 4});

		WHEN("input grid minimum point is mapped to source grid")
		{
			felt2::Vec3i const input_pos{0, 0, 0};
			felt2::Vec3i const source_pos =
				FilterSizeHelper<>::source_pos_from_input_pos_and_filter_size(
					filter_size, filter_stride, input_pos);

			THEN("source pos is at minimum of source grid")
			{
				CHECK(source_pos == felt2::Vec3i{0, 0, 0});
			}
		}

		WHEN("input grid maximum point is mapped to source grid")
		{
			felt2::Vec3i const input_pos{5, 3, 3};
			felt2::Vec3i const source_pos =
				FilterSizeHelper<>::source_pos_from_input_pos_and_filter_size(
					filter_size, filter_stride, input_pos);

			THEN("source pos is at maximum of source grid")
			{
				CHECK(source_pos == felt2::Vec3i{3, 3, 3});
			}
		}

		WHEN("source grid minimum point is mapped to input grid")
		{
			felt2::Vec3i const source_pos{0, 0, 0};

			using PosArray = std::vector<felt2::Vec3i>;
			PosArray input_pos_list;
			FilterSizeHelper<>::input_pos_from_source_pos_and_filter_size(
				filter_size,
				filter_stride,
				source_size,
				source_pos,
				[&](felt2::Vec3i const & pos_) { input_pos_list.push_back(pos_); });

			THEN("source point maps to a single input grid point at the minimum of the input grid")
			{
				CAPTURE(input_pos_list);
				CHECK(std::ranges::equal(input_pos_list, PosArray{felt2::Vec3i{0, 0, 0}}));
			}
		}

		WHEN("source grid maximum point is mapped to input grid")
		{
			felt2::Vec3i const source_pos{3, 3, 3};

			using PosArray = std::vector<felt2::Vec3i>;
			PosArray input_pos_list;
			FilterSizeHelper<>::input_pos_from_source_pos_and_filter_size(
				filter_size,
				filter_stride,
				source_size,
				source_pos,
				[&](felt2::Vec3i const & pos_) { input_pos_list.push_back(pos_); });

			THEN("source point maps to a single input grid point at the maximum of the input grid")
			{
				CAPTURE(input_pos_list);
				CHECK(std::ranges::equal(input_pos_list, PosArray{felt2::Vec3i{5, 3, 3}}));
			}
		}

		WHEN("source grid intermediate point is mapped to input grid")
		{
			felt2::Vec3i const source_pos{1, 1, 1};

			using PosArray = std::vector<felt2::Vec3i>;
			PosArray input_pos_list;
			FilterSizeHelper<>::input_pos_from_source_pos_and_filter_size(
				filter_size,
				filter_stride,
				source_size,
				source_pos,
				[&](felt2::Vec3i const & pos_) { input_pos_list.push_back(pos_); });

			THEN("source point maps to 2 separate filter inputs")
			{
				CHECK(input_pos_list == PosArray{felt2::Vec3i{1, 1, 1}, felt2::Vec3i{2, 1, 1}});
			}
		}

		WHEN("source grid strided point is mapped to input grid")
		{
			felt2::Vec3i const source_pos{0, 2, 0};

			using PosArray = std::vector<felt2::Vec3i>;
			PosArray filter_pos_list;
			PosArray global_pos_list;
			FilterSizeHelper<>::input_pos_from_source_pos_and_filter_size(
				filter_size,
				filter_stride,
				source_size,
				source_pos,
				[&](felt2::Vec3i const & filter_pos_, felt2::Vec3i const & global_pos_)
				{
					filter_pos_list.push_back(filter_pos_);
					global_pos_list.push_back(global_pos_);
				});

			THEN("source point maps to a single position in a filter input")
			{
				CAPTURE(filter_pos_list);
				CAPTURE(global_pos_list);
				CHECK(std::ranges::equal(filter_pos_list, PosArray{felt2::Vec3i{0, 1, 0}}));
				CHECK(std::ranges::equal(global_pos_list, PosArray{felt2::Vec3i{0, 2, 0}}));
			}
		}
	}
}
