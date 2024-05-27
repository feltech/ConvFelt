// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/FilterSizeHelper.hpp>
#include <convfelt/felt2/typedefs.hpp>
#include <convfelt/iter.hpp>

SCENARIO("Input/output ConvGrids")
{
	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{k_file_path}};
		CAPTURE(image.geterror(false));
		REQUIRE(!image.has_error());
		image.read();

		convfelt::InputGrid image_grid{
			convfelt::make_host_context(),
			convfelt::InputGrid::PowTwoDu::from_minimum_size(
				{image.spec().height + 2, image.spec().width + 2, image.spec().nchannels}),
			{0, 0, 0},
			0};

		auto image_grid_spec = image.spec();
		image_grid_spec.width = image_grid.size().as_pos()(1);
		image_grid_spec.height = image_grid.size().as_pos()(0);
		image_grid_spec.nchannels = image_grid.size().as_pos()(2);
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		WHEN("image is split into filter regions")
		{
			// Helper to convert sizes and positions between source image, filter input grid, and
			// output image.
			FilterSizeHelper const filter_input_sizer{
				felt2::PowTwo3u::from_exponents({2, 2, 2}), felt2::Vec3i{2, 2, 0}};

			// Grid of filter inputs expanded from source image, with child grids of input values
			// for each filter side-by-side. Typically the input grid will be much larger than the
			// source image grid, since many pixels are duplicated, depending on the stride of the
			// filter as it moves across the source image.
			convfelt::ConvGrid filter_input_grid{
				convfelt::make_host_context(),
				filter_input_sizer.input_size_from_source_size(image_grid.size()),
				filter_input_sizer.filter_size};

			// Each child grid of the input grid represents the input to a filter.
			CHECK(filter_input_grid.child_size() == filter_input_sizer.filter_size);

			// Loop each individual filter input child.
			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				// Calculate position in original source image that corresponds to the local (0,0)
				// point in the filter.
				felt2::Vec3i const source_pos_start =
					filter_input_sizer.source_start_pos_from_filter_pos(
						filter_input_grid.children().index(filter_pos_idx));

				// Loop each position within the filter.
				for (felt2::PosIdx const local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					// Calculate the position in the original source image that the current filter
					// pixel corresponds to.
					felt2::Vec3i const source_pos =
						source_pos_start + felt2::index(local_pos_idx, filter.size());

					// source_pos may be out of bounds wrt image_grid because the filter_input_grid
					// can have additional padding due to pow2 sizing.
					if (!image_grid.inside(source_pos))
						continue;

					// Set the filter input value from the corresponding value in the source image.
					filter.set(local_pos_idx, image_grid.get(source_pos));
				}
			}

			THEN("regions have expected values")
			{
				{
					auto const & filter_input = filter_input_grid.children().get({0, 0, 0});
					CHECK(filter_input.size() == filter_input_sizer.filter_size);

					for (auto const & filter_grid_pos : convfelt::iter::pos(filter_input))
					{
						CAPTURE(filter_grid_pos);
						CHECK(filter_input.get(filter_grid_pos) == image_grid.get(filter_grid_pos));
					}
				}
				{
					// Check that each filter input pixel corresponds to the expected source image
					// pixel.
					std::size_t num_nonzero = 0;
					for (auto const & filter_input :
						 convfelt::iter::val(filter_input_grid.children()))
					{
						CHECK(filter_input.size() == filter_input_sizer.filter_size);

						for (auto const & [pos_idx, pos] :
							 convfelt::iter::idx_and_pos(filter_input))
						{
							felt2::Vec3i const source_pos =
								filter_input_sizer.source_pos_from_input_pos(pos);

							CAPTURE(pos);
							CAPTURE(source_pos);

							// source_pos might be outside domain of image_grid since pow2 sizing
							// means filter_input_grid is likely larger than necessary.
							if (!image_grid.inside(source_pos))
								CHECK(filter_input.get(pos_idx) == 0);
							else
								CHECK(filter_input.get(pos_idx) == image_grid.get(source_pos));

							if (filter_input.get(pos_idx) != 0)
								++num_nonzero;
						}
					}

					CHECK(num_nonzero > 0);

					// Check that each source image pixel maps to the appropriate filter input
					// pixel.
					for (auto const & [pos_idx, pos] : convfelt::iter::idx_and_pos(image_grid))
					{
						filter_input_sizer.input_pos_from_source_pos(
							image_grid.size().as_pos(),
							pos,
							[&](auto const filter_pos, auto const input_pos)
							{
								felt2::Vec3i const source_pos =
									filter_input_sizer.source_pos_from_input_pos(input_pos);

								CHECK(
									filter_input_grid.children().get(filter_pos).get(input_pos) ==
									image_grid.get(source_pos));
							});
					}
				}
			}
		}
	}
}
