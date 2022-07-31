#include <CL/sycl.hpp>
#ifdef SYCL_DEVICE_ONLY
#undef SYCL_DEVICE_ONLY
#endif
#include <Eigen/Eigen>
#include <catch2/catch.hpp>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/meta/result_of.hpp>
#include <viennacl/vector.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/Numeric.hpp>
#include <convfelt/iter.hpp>
#ifdef __CUDACC__
#ifndef __CUDA_ARCH__
#undef __CUDACC__
#endif
#endif
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
namespace viennacl
{
namespace result_of
{
template <typename PlainObjectType, int MapOptions, typename StrideType>
struct size_type<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
{
	using type = Eigen::Index;
};
}  // namespace result_of

namespace traits
{
template <typename PlainObjectType, int MapOptions, typename StrideType>
inline std::size_t size2(Eigen::Map<PlainObjectType, MapOptions, StrideType> const & mat)
{
	return static_cast<std::size_t>(mat.cols());
}

template <typename PlainObjectType, int MapOptions, typename StrideType>
inline std::size_t size1(Eigen::Map<PlainObjectType, MapOptions, StrideType> const & mat)
{
	return static_cast<std::size_t>(mat.rows());
}
}  // namespace traits
}  // namespace viennacl

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

	viennacl::copy(lhs, vcl_lhs);
	viennacl::copy(rhs, vcl_rhs);

	vcl_result = vcl_lhs + vcl_rhs;

	viennacl::copy(vcl_result, result);

	CHECK(result == Eigen::VectorXd::Constant(100, 2));
}

SCENARIO("Using ViennaCL with Felt")
{
	GIVEN("a grid partitioned by filter size with non-uniform values in first partition")
	{
		constexpr Felt::Dim filter_dim = 3;
		const Felt::Vec2i filter_dims{filter_dim, filter_dim};
		constexpr Felt::Dim filter_size = filter_dim * filter_dim * filter_dim;

		// 5D grid.
		using Grid = convfelt::ConvGrid;

		// Imagine a 126x126x3 (RGB) image with 42x 3x3 convolutions, all white.
		Grid grid{{126, 126, 3}, filter_dims};
		grid.set({0, 0, 0}, 2);
		grid.set({0, 0, grid.child_size().x() - 1}, 3);
		grid.set({0, grid.child_size().y() - 1, 0}, 4);
		grid.set({grid.child_size().z() - 1, 0, 0}, 5);

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

				viennacl::copy(grid.children().get({0, 0, 0}).matrix(), vcl_input);
				viennacl::copy(filter, vcl_filter);

				vcl_result = viennacl::linalg::prod(vcl_filter, vcl_input);

				auto result = grid.children().get({0, 0, 0}).matrix();
				viennacl::copy(vcl_result, result);

				THEN("output data has expected values")
				{
					Eigen::VectorXf expected = Eigen::VectorXf::Constant(filter_size, 2);
					expected(0) = 4;

					auto child = grid.children().get({0, 0, 0});

					CHECK(child.get({0, 0, 0}) == 4);
					CHECK(child.get({0, 0, filter_dim - 1}) == 6);
					CHECK(child.get({0, filter_dim - 1, 0}) == 8);
					CHECK(child.get({filter_dim - 1, 0, 0}) == 10);
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

			FilterGrid filter_input_grid = [&image_grid, filter_stride]
			{
				Felt::Vec2i const filter_input_window{4, 4};
				Felt::Vec3i const filter_input_shape{4, 4, 3};

				Felt::Vec3i num_filters = (image_grid.size() - filter_input_shape);
				num_filters = num_filters / filter_stride + Felt::Vec3i::Ones();
				Felt::Vec3i const num_connections =
					(num_filters.array() * filter_input_shape.array()).matrix();

				return FilterGrid{num_connections, filter_input_window};
			}();
			const Felt::Vec3i filter_input_shape = filter_input_grid.child_size();

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				const Felt::Vec3i input_pos_start =
					filter_input_grid.children().index(filter_pos_idx) * filter_stride;
				(void)input_pos_start;

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

				convfelt::ConvGrid filter_output_grid{filter_output_grid_size, filter_output_shape};

				CHECK(filter_output_grid.children().size() == filter_input_grid.children().size());

				Eigen::MatrixXf weights{filter_output_shape.prod(), filter_input_shape.prod()};
				weights.setConstant(1);

				auto const check_output = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid.children()))
					{
						convfelt::Scalar const expected =
							filter_input_grid.children().get(filter_pos_idx).matrix().sum();

						const Felt::Vec3i filter_pos =
							filter_input_grid.children().index(filter_pos_idx);
						CAPTURE(filter_pos);

						for (auto const & actual :
							 convfelt::iter::val(filter_output_grid.children().get(filter_pos_idx)))
						{
							CHECK(actual == expected);
						}
					}
				};

				WHEN("weight are applied to input to produce output filter-by-filter")
				{
					using vclvec = viennacl::vector<convfelt::Scalar>;
					using vclmat = viennacl::matrix<convfelt::Scalar>;
					vclvec vcl_input{static_cast<vclvec::size_type>(filter_input_shape.prod())};
					vclvec vcl_result{static_cast<vclvec::size_type>(filter_output_shape.prod())};
					vclmat vcl_weights{
						static_cast<vclvec::size_type>(weights.rows()),
						static_cast<vclvec::size_type>(weights.cols())};
					viennacl::copy(weights, vcl_weights);

					for (auto const & pos_idx :
						 convfelt::iter::pos_idx(filter_input_grid.children()))
					{
						viennacl::copy(
							filter_input_grid.children().get(pos_idx).matrix(), vcl_input);

						vcl_result = viennacl::linalg::prod(vcl_weights, vcl_input);

						auto output_array = filter_output_grid.children().get(pos_idx).matrix();
						viennacl::copy(vcl_result, output_array);
					}

					THEN("output data has expected values")
					{
						check_output();
					}
				}  // WHEN("weight are applied to input to produce output filter-by-filter")

				WHEN("weight are applied to input to produce output all at once")
				{
					using OclInput = viennacl::matrix<convfelt::Scalar, viennacl::column_major>;
					using OclWeights = viennacl::matrix<convfelt::Scalar, viennacl::row_major>;
					auto input_rows = static_cast<OclInput::size_type>(filter_input_shape.prod());
					auto cols = static_cast<OclInput::size_type>(
						filter_input_grid.children().size().prod());
					auto output_rows = static_cast<OclInput::size_type>(filter_output_shape.prod());

					OclInput vcl_input_all{input_rows, cols};
					OclInput vcl_result_all{output_rows, cols};

					OclWeights vcl_weights{
						static_cast<OclInput::size_type>(weights.rows()),
						static_cast<OclInput::size_type>(weights.cols())};
					viennacl::copy(weights, vcl_weights);

					viennacl::fast_copy(
						&filter_input_grid.data()[0],
						&filter_input_grid.data()[filter_input_grid.data().size()],
						vcl_input_all);

					vcl_result_all = viennacl::linalg::prod(vcl_weights, vcl_input_all);

					viennacl::fast_copy(vcl_result_all, &filter_output_grid.data()[0]);

					THEN("output data has expected values")
					{
						check_output();
					}
				}  // WHEN("weight are applied to input to produce output all at once")
			}
		}
	}
}

SCENARIO("Basic SyCL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f};
		std::vector<float> b = {-1.f, 2.f, -3.f, 4.f, -5.f};
		std::vector<float> c(a.size());
		assert(a.size() == b.size());

		WHEN("vectors are added using sycl")
		{
			{
				cl::sycl::gpu_selector selector;
				cl::sycl::queue q{selector};
				cl::sycl::range<1> work_items{a.size()};
				cl::sycl::buffer<float> buff_a(a.data(), a.size());
				cl::sycl::buffer<float> buff_b(b.data(), b.size());
				cl::sycl::buffer<float> buff_c(c.data(), c.size());

				q.submit(
					[&](cl::sycl::handler & cgh)
					{
						auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
						auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
						auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

						cgh.parallel_for<class vector_add>(
							work_items,
							[=](cl::sycl::id<1> tid)
							{ access_c[tid] = access_a[tid] + access_b[tid]; });
					});
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {0.f, 4.f, 0.f, 8.f, 0.f};

				CHECK(c == expected);
			}
		}
	}
}