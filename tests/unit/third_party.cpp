#include <catch2/catch.hpp>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <Eigen/Eigen>
#include <Felt/Impl/Grid.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/when_all_ready.hpp>
#include <viennacl/device_specific/builtin_database/common.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/Numeric.hpp>

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
		using Grid = ConvFelt::ConvGrid<float, 3>;

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

SCENARIO("OpenImageIO with bit of cppcoro and Felt")
{
	GIVEN("a simple monochrome image file")
	{
		static constexpr std::string_view file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		using Pixel = Felt::VecDT<float, 3>;

		WHEN("file is loaded")
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

				WHEN("image is loaded into grid with no zero-padding")
				{
					auto image_grid_spec = image.spec();
					image_grid_spec.format = OIIO::TypeDescFromC<convfelt::Scalar>::value();

					Felt::Impl::Grid::Simple<convfelt::Scalar, 3> image_grid{
						{image_grid_spec.height, image_grid_spec.width, image_grid_spec.nchannels},
						{0, 0, 0},
						0};

					OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
					OIIO::ImageBufAlgo::paste(image_grid_buf, 0, 0, 0, 0, image);

					THEN("grid contains image")
					{
						for (Felt::NodeIdx x = 0; x < image_grid.size().x(); ++x)
							for (Felt::NodeIdx y = 0; y < image_grid.size().y(); ++y)
							{
								Pixel pixel;
								image.getpixel(y, x, pixel.data(), 3);
								for (Felt::NodeIdx z = 0; z < image_grid.size().z(); ++z)
								{
									CAPTURE(x, y, z);
									CHECK(image_grid.get({x, y, z}) == pixel(z));
								}
							}
					}
				}

				WHEN("image is loaded into grid with 1 pixel of zero-padding")
				{
					auto image_grid_spec = image.spec();
//					image_grid_spec.x = 1;
//					image_grid_spec.y = 1;
//					image_grid_spec.full_x = 1;
//					image_grid_spec.full_y = 1;
//					image_grid_spec.full_width += 2;
//					image_grid_spec.full_height += 2;
					image_grid_spec.width += 2;
					image_grid_spec.height += 2;
					image_grid_spec.format = OIIO::TypeDescFromC<convfelt::Scalar>::value();

					Felt::Impl::Grid::Simple<convfelt::Scalar, 3> image_grid{
						{image.spec().height + 2,
						 image.spec().width + 2,
						 image_grid_spec.nchannels},
						{0, 0, 0},
						0};

					OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
//					image_grid_buf.copy_pixels(image);
					OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

					THEN("grid contains padded image")
					{
						for (Felt::NodeIdx x = 0; x < image_grid.size().x(); ++x)
							for (Felt::NodeIdx y = 0; y < image_grid.size().y(); ++y)
							{
								if (x == 0 || y == 0 ||
									//
									x == image_grid.size().x() - 1 ||
									y == image_grid.size().y() - 1)
								{
									for (Felt::NodeIdx z = 0; z < image_grid.size().z(); ++z)
									{
										CAPTURE(x, y, z);
										CHECK(image_grid.get({x, y, z}) == 0);
									}
								}
								else
								{
									Pixel pixel;
									image.getpixel(y - 1, x - 1, pixel.data(), 3);
									for (Felt::NodeIdx z = 0; z < image_grid.size().z(); ++z)
									{
										CAPTURE(x, y, z);
										CHECK(image_grid.get({x, y, z}) == pixel(z));
									}
								}
							}

						CHECK(image_grid.get({1, 64, 0}) == 1);
						CHECK(image_grid.get({64, 0, 0}) == 0);
						CHECK(image_grid.get({64, 0, 1}) == 0);
						CHECK(image_grid.get({64, 0, 2}) == 0);
						CHECK(image_grid.get({64, 1, 0}) == 1);
						CHECK(image_grid.get({64, 1, 1}) == 1);
						CHECK(image_grid.get({64, 1, 2}) == 1);
					}
				}  // WHEN("image is loaded into grid with 1 pixel of zero-padding")
			}  // THEN("file has expected properties")
		}
	}
}