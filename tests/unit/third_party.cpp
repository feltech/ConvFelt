#include <ranges>

#include <catch2/catch.hpp>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <Eigen/Eigen>
#include <Felt/Impl/Grid.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <viennacl/device_specific/builtin_database/common.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/Numeric.hpp>
#include <convfelt/iter.hpp>

SCENARIO("Loading ViennaCL")
{
	/**
	 *  Retrieve the platforms and iterate:
	 **/
	std::vector<viennacl::ocl::platform> platforms = viennacl::ocl::get_platforms();

	REQUIRE(!platforms.empty());

	bool is_first_element = true;
	for (auto & platform : platforms)
	{
		std::vector<viennacl::ocl::device> devices = platform.devices(CL_DEVICE_TYPE_ALL);

		/**
		 *  Print some platform information
		 **/
		std::cout << "# =========================================" << std::endl;
		std::cout << "#         Platform Information             " << std::endl;
		std::cout << "# =========================================" << std::endl;

		std::cout << "#" << std::endl;
		std::cout << "# Vendor and version: " << platform.info() << std::endl;
		std::cout << "#" << std::endl;

		if (is_first_element)
		{
			std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
			is_first_element = false;
		}

		/*
		 * Traverse the devices and print all information available using the convenience member
		 * function full_info():
		 */
		std::cout << "# " << std::endl;
		std::cout << "# Available Devices: " << std::endl;
		std::cout << "# " << std::endl;
		for (auto & device : devices)
		{
			std::cout << std::endl;

			std::cout << "  -----------------------------------------" << std::endl;
			std::cout << device.full_info();
			std::cout << "ViennaCL Device Architecture:  " << device.architecture_family()
					  << std::endl;
			std::cout << "ViennaCL Database Mapped Name: "
					  << viennacl::device_specific::builtin_database::get_mapped_device_name(
							 device.name(), device.vendor_id())
					  << std::endl;
			std::cout << "  -----------------------------------------" << std::endl;
		}
		std::cout << std::endl;
		std::cout << "###########################################" << std::endl;
		std::cout << std::endl;
	}
}

SCENARIO("Using ViennaCL with Eigen")
{
	Eigen::VectorXd lhs(100);
	lhs.setOnes();
	Eigen::VectorXd rhs(100);
	rhs.setOnes();
	Eigen::VectorXd result(100);
	result.setZero();

	viennacl::vector<double> vcl_lhs(100);
	viennacl::vector<double> vcl_rhs(100);
	viennacl::vector<double> vcl_result(100);

	viennacl::copy(lhs.begin(), lhs.end(), vcl_lhs.begin());
	viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());

	vcl_result = vcl_lhs + vcl_rhs;

	viennacl::copy(vcl_result.begin(), vcl_result.end(), result.begin());

	CHECK(result == Eigen::VectorXd::Constant(100, 2));
}

SCENARIO("Using ViennaCL with Felt")
{
	GIVEN("a grid partitioned by filter size with non-uniform values in first partition")
	{
		constexpr Felt::Dim filter_dim = 3;
		const Felt::Vec3i filter_dims{filter_dim, filter_dim, filter_dim};
		constexpr Felt::Dim filter_size = filter_dim * filter_dim * filter_dim;

		// 5D grid.
		using Grid = convfelt::ConvGrid;

		// Imagine a 126x126x3 (RGB) image with 42x 3x3 convolutions, all white.
		Grid grid{{126, 126, 3}, {0, 0, 0}, filter_dims, 1.0f};
		grid.set({0, 0, 0}, 2);

		AND_GIVEN("a filter matrix that scales inputs by 2")
		{
			// Convolution matrix to scale input by x2.
			Eigen::MatrixXf filter{filter_size, filter_size};
			filter.setIdentity();
			filter *= 2;

			WHEN("scaling is applied to first filter partition using OpenCL")
			{
				viennacl::vector<float> vcl_input{filter_size};
				viennacl::vector<float> vcl_result{filter_size};
				viennacl::matrix<float> vcl_filter{filter_size, filter_size};

				viennacl::copy(grid.children().get({0, 0, 0}).array(), vcl_input);
				viennacl::copy(filter, vcl_filter);

				vcl_result = viennacl::linalg::prod(vcl_filter, vcl_input);

				auto result = grid.children().get({0, 0, 0}).matrix();
				viennacl::copy(vcl_result, result);

				THEN("output data has expected values")
				{
					Eigen::VectorXf expected = Eigen::VectorXf::Constant(filter_size, 2);
					expected(0) = 4;

					auto actual = grid.children().get({0, 0, 0}).matrix();

					CHECK(actual == expected);
				}
			}
		}
	}
}

SCENARIO("Using OpenImageIO with cppcoro and loading into Felt grid")
{
	GIVEN("a simple monochrome image file")
	{
		static constexpr std::string_view file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		using Pixel = Felt::VecDT<float, 3>;

		WHEN("image file is loaded")
		{
			auto task = []() -> cppcoro::task<OIIO::ImageBuf>
			{
				OIIO::ImageBuf imageBuf{std::string{file_path}};
				imageBuf.read();
				co_return imageBuf;
			};

			OIIO::ImageBuf image = cppcoro::sync_wait(task());

			THEN("file has expected properties")
			{
				CHECK(image.spec().height == 128);
				CHECK(image.spec().width == 128);
				CHECK(image.spec().nchannels == 3);
				{
					Pixel pixel;
					image.getpixel(0, 0, pixel.data(), 3);
					CHECK(pixel == Pixel{0, 0, 0});
					image.getpixel(0, 127, pixel.data(), 3);
					CHECK(pixel == Pixel{0, 0, 0});
					image.getpixel(127, 0, pixel.data(), 3);
					CHECK(pixel == Pixel{0, 0, 0});
					image.getpixel(127, 127, pixel.data(), 3);
					CHECK(pixel == Pixel{0, 0, 0});

					image.getpixel(64, 0, pixel.data(), 3);
					CHECK(pixel == Pixel{1, 1, 1});
					image.getpixel(127, 64, pixel.data(), 3);
					CHECK(pixel == Pixel{1, 1, 1});
					image.getpixel(64, 127, pixel.data(), 3);
					CHECK(pixel == Pixel{1, 1, 1});
					image.getpixel(0, 64, pixel.data(), 3);
					CHECK(pixel == Pixel{1, 1, 1});
				}
			}

			AND_WHEN("image is loaded into grid with no zero-padding")
			{
				auto image_grid_spec = image.spec();
				image_grid_spec.format = OIIO::TypeDescFromC<convfelt::Scalar>::value();

				convfelt::InputGrid image_grid{
					{image_grid_spec.height, image_grid_spec.width, image_grid_spec.nchannels},
					{0, 0, 0},
					0};

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
				OIIO::ImageBufAlgo::paste(image_grid_buf, 0, 0, 0, 0, image);

				THEN("grid contains image")
				{
					for (auto x : convfelt::iter::idx(image_grid.size().x()))
						for (auto y : convfelt::iter::idx(image_grid.size().y()))
						{
							Pixel pixel;
							image.getpixel(y, x, pixel.data(), 3);
							for (auto z : convfelt::iter::idx(image_grid.size().z()))
							{
								CAPTURE(x, y, z);
								CHECK(image_grid.get({x, y, z}) == pixel(z));
							}
						}
				}
			}

			AND_WHEN("image is loaded into grid with 1 pixel of zero-padding")
			{
				auto image_grid_spec = image.spec();
				image_grid_spec.width += 2;
				image_grid_spec.height += 2;
				image_grid_spec.format = OIIO::TypeDescFromC<convfelt::Scalar>::value();

				convfelt::InputGrid image_grid{
					{image.spec().height + 2, image.spec().width + 2, image_grid_spec.nchannels},
					{0, 0, 0},
					0};

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
				OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

				THEN("grid contains padded image")
				{
					for (auto x : convfelt::iter::idx(image_grid.size().x()))
						for (auto y : convfelt::iter::idx(image_grid.size().y()))
						{
							if (x == 0 || y == 0 ||
								//
								x == image_grid.size().x() - 1 || y == image_grid.size().y() - 1)
							{
								for (auto z : convfelt::iter::idx(image_grid.size().z()))
								{
									CAPTURE(x, y, z);
									CHECK(image_grid.get({x, y, z}) == 0);
								}
							}
							else
							{
								Pixel pixel;
								image.getpixel(y - 1, x - 1, pixel.data(), 3);
								for (auto z : convfelt::iter::idx(image_grid.size().z()))
								{
									CAPTURE(x, y, z);
									CHECK(image_grid.get({x, y, z}) == pixel(z));
								}
							}
						}
				}
			}
		}
	}
}

SCENARIO("Input/output ConvGrids")
{
	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{file_path}};
		image.read();
		auto image_grid_spec = image.spec();
		image_grid_spec.width += 2;
		image_grid_spec.height += 2;
		image_grid_spec.format = OIIO::TypeDescFromC<convfelt::Scalar>::value();

		convfelt::InputGrid image_grid{
			{image.spec().height + 2, image.spec().width + 2, image_grid_spec.nchannels},
			{0, 0, 0},
			0};

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		WHEN("image is split into filter regions")
		{
			using FilterGrid = convfelt::ConvGrid;

			const Felt::NodeIdx filter_stride = 2;

			const Felt::Vec3i filter_input_shape{4, 4, image_grid.size().z()};
			FilterGrid filter_input_grid = [&image_grid, &filter_input_shape, filter_stride]
			{
				const Felt::Vec3i num_filters =
					(image_grid.size() - filter_input_shape) / filter_stride + Felt::Vec3i::Ones();
				const Felt::Vec3i num_connections =
					(num_filters.array() * filter_input_shape.array()).matrix();

				return FilterGrid{num_connections, {0, 0, 0}, filter_input_shape, 0};
			}();

			for (auto [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				const Felt::Vec3i input_pos_start =
					filter_input_grid.children().index(filter_pos_idx) * filter_stride;

				for (Felt::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const Felt::Vec3i input_pos =
						input_pos_start + Felt::index<3>(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
				}
			}

			THEN("regions have expected values")
			{
				{
					auto const & filter_input = filter_input_grid.children().get({0, 0, 0});
					CHECK(filter_input.size() == filter_input_shape);

					for (auto const & filter_grid_pos : convfelt::iter::pos(filter_input))
					{
						CAPTURE(filter_grid_pos);
						CHECK(filter_input.get(filter_grid_pos) == image_grid.get(filter_grid_pos));
					}
				}
				{
					std::size_t num_nonzero = 0;
					for (auto const & filter_input :
						 convfelt::iter::val(filter_input_grid.children()))
					{
						CHECK(filter_input.size() == filter_input_shape);

						for (auto const & [pos_idx, pos] :
							 convfelt::iter::idx_and_pos(filter_input))
						{
							Felt::Vec3i const filter_image_start_pos =
								filter_stride * (pos.array() / filter_input_shape.array()).matrix();

							Felt::Vec3i const image_grid_pos = pos - filter_image_start_pos;

							CAPTURE(pos);
							CAPTURE(image_grid_pos);
							CHECK(filter_input.get(pos_idx) == image_grid.get(image_grid_pos));

							if (filter_input.get(pos) != 0)
								++num_nonzero;
						}
					}
					CHECK(num_nonzero > 0);
				}
			}

			AND_GIVEN("an output grid and weight matrix")
			{
				Felt::Vec3i const filter_output_shape{2, 2, 4};
				Felt::Vec3i const filter_output_grid_size =
					(filter_input_grid.children().size().array() * filter_output_shape.array())
						.matrix();

				convfelt::ConvGrid filter_output_grid{
					filter_output_grid_size, {0, 0, 0}, filter_output_shape, 0};

				CHECK(filter_output_grid.children().size() == filter_input_grid.children().size());

				Eigen::MatrixXf weights{filter_output_shape.prod(), filter_input_shape.prod()};
				weights.setConstant(1);

				WHEN("weight are applied to input to produce output")
				{
					using vclvec = viennacl::vector<convfelt::Scalar>;
					using vclmat = viennacl::matrix<convfelt::Scalar>;

					vclvec vcl_input(static_cast<vclvec::size_type>(filter_input_shape.prod()));
					vclvec vcl_result(static_cast<vclvec::size_type>(filter_output_shape.prod()));
					vclmat vcl_weights{
						static_cast<vclvec::size_type>(weights.rows()),
						static_cast<vclvec::size_type>(weights.cols())};
					viennacl::copy(weights, vcl_weights);

					for (auto const & pos_idx :
						 convfelt::iter::pos_idx(filter_input_grid.children()))
					{
						viennacl::copy(
							filter_input_grid.children().get(pos_idx).array(), vcl_input);

						vcl_result = viennacl::linalg::prod(vcl_weights, vcl_input);

						auto output_array = filter_output_grid.children().get(pos_idx).matrix();
						viennacl::copy(vcl_result, output_array);
					}

					THEN("output data has expected values")
					{
						for (auto const & filter_pos_idx :
							 convfelt::iter::pos_idx(filter_output_grid.children()))
						{
							convfelt::Scalar const expected =
								filter_input_grid.children().get(filter_pos_idx).matrix().sum();

							const Felt::Vec3i filter_pos =
								filter_input_grid.children().index(filter_pos_idx);
							CAPTURE(filter_pos);

							for (auto const & actual : convfelt::iter::val(
									 filter_output_grid.children().get(filter_pos_idx)))
							{
								CHECK(actual == expected);
							}
						}
					}
				}
			}
		}
	}
}