// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>

#include <Eigen/Core>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/iter.hpp>

SCENARIO("Using OpenImageIO with cppcoro and loading into grid")
{
	GIVEN("a simple monochrome image file")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		using Pixel = Eigen::VectorXf;

		WHEN("image file is loaded")
		{
			auto task = []() -> cppcoro::task<OIIO::ImageBuf>
			{
				OIIO::ImageBuf image_buf{std::string{k_file_path}};
				image_buf.read();
				co_return image_buf;
			};

			OIIO::ImageBuf const image = cppcoro::sync_wait(task());

			CAPTURE(image.geterror(false));
			REQUIRE(!image.has_error());

			THEN("file has expected properties")
			{
				CHECK(image.spec().height == 128);
				CHECK(image.spec().width == 128);
				CHECK(image.spec().nchannels == 3);
				{
					Pixel pixel(image.spec().nchannels);
					image.getpixel(0, 0, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 0));
					image.getpixel(0, 127, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 0));
					image.getpixel(127, 0, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 0));
					image.getpixel(127, 127, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 0));

					image.getpixel(64, 0, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 1));
					image.getpixel(127, 64, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 1));
					image.getpixel(64, 127, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 1));
					image.getpixel(0, 64, pixel.data(), static_cast<int>(pixel.size()));
					CHECK(pixel == Pixel::Constant(image.spec().nchannels, 1));
				}
			}

			AND_WHEN("image is loaded into grid with no zero-padding")
			{
				convfelt::InputGrid image_grid{
					convfelt::make_host_context(),
					convfelt::InputGrid::PowTwoDu::from_minimum_size(
						{image.spec().height, image.spec().width, image.spec().nchannels}),
					{0, 0, 0},
					0};

				auto image_grid_spec = image.spec();
				image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();
				image_grid_spec.width = image_grid.size().as_pos().y();
				image_grid_spec.height = image_grid.size().as_pos().x();
				image_grid_spec.nchannels = image_grid.size().as_pos().z();

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
				OIIO::ImageBufAlgo::paste(image_grid_buf, 0, 0, 0, 0, image);

				// std::string tmp_img_file_path = std::tmpnam(nullptr);  // NOLINT(*-mt-unsafe)
				// tmp_img_file_path += ".png";
				// image_grid_buf.write(tmp_img_file_path, OIIO::TypeColor);
				// CAPTURE(tmp_img_file_path);

				THEN("grid contains image")
				{
					for (auto x : convfelt::iter::idx(image_grid.size().as_pos().x()))
						for (auto y : convfelt::iter::idx(image_grid.size().as_pos().y()))
						{
							// Grid can be larger than image since pow2 size.
							if (x >= image_grid_spec.height || y >= image_grid_spec.width)
							{
								for (auto z : convfelt::iter::idx(image_grid.size().as_pos().z()))
								{
									CAPTURE(x, y, z);
									CHECK(image_grid.get({x, y, z}) == 0);
								}
							}
							else
							{
								Pixel pixel(image.spec().nchannels);
								image.getpixel(y, x, pixel.data(), static_cast<int>(pixel.size()));
								for (auto z : convfelt::iter::idx(image_grid.size().as_pos().z()))
								{
									CAPTURE(x, y, z);
									// Grid can have more channels than source because pow2 size.
									if (z >= image.spec().nchannels)
									{
										CHECK(image_grid.get({x, y, z}) == 0);
									}
									else
									{
										CHECK(image_grid.get({x, y, z}) == pixel(z));
									}
								}
							}
						}
				}
			}

			AND_WHEN("image is loaded into grid with 1 pixel of zero-padding")
			{
				convfelt::InputGrid image_grid{
					convfelt::make_host_context(),
					convfelt::InputGrid::PowTwoDu::from_minimum_size(
						{image.spec().height + 2, image.spec().width + 2, image.spec().nchannels}),
					{0, 0, 0},
					0};

				auto image_grid_spec = image.spec();
				image_grid_spec.width = image_grid.size().as_pos().y();
				image_grid_spec.height = image_grid.size().as_pos().x();
				image_grid_spec.nchannels = image_grid.size().as_pos().z();
				image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
				OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

				THEN("grid contains padded image")
				{
					for (auto x : convfelt::iter::idx(image_grid.size().as_pos().x()))
						for (auto y : convfelt::iter::idx(image_grid.size().as_pos().y()))
						{
							if (x == 0 || y == 0 ||
								//
								x >= image.spec().height + 1 || y >= image.spec().height + 1)
							{
								for (auto z : convfelt::iter::idx(image_grid.size().as_pos().z()))
								{
									CAPTURE(x, y, z);
									CHECK(image_grid.get({x, y, z}) == 0);
								}
							}
							else
							{
								Pixel pixel(image.spec().nchannels);
								image.getpixel(
									y - 1, x - 1, pixel.data(), static_cast<int>(pixel.size()));
								for (auto z : convfelt::iter::idx(image_grid.size().as_pos().z()))
								{
									CAPTURE(x, y, z);
									// Grid can have more channels than source because pow2 size.
									if (z >= image.spec().nchannels)
									{
										CHECK(image_grid.get({x, y, z}) == 0);
									}
									else
									{
										CHECK(image_grid.get({x, y, z}) == pixel(z));
									}
								}
							}
						}
				}
			}
		}
	}
}