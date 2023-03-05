#include <span>

#include <sycl/sycl.hpp>
namespace sycl
{
// Required for MKL.
template <typename... Args>
using span = std::span<Args...>;
}  // namespace sycl
#include <oneapi/mkl.hpp>

// Eigen
//   * OpenSYCL (hipSYCL) not supported because missing `isinf` and `isfinite` builtins. So must
//     #undef SYCL_DEVICE_ONLY (which probably shouldn't be set by OpenSYCL, but it is).
//   * Luckily CUDA/HIP are also supported.
//     TODO(DF): but is the CUDA/HIP optimized code path actually used in the device kernels?
// Required for Eigen.
#ifdef SYCL_DEVICE_ONLY
#define was_SYCL_DEVICE_ONLY SYCL_DEVICE_ONLY
#undef SYCL_DEVICE_ONLY
#endif
#include <Eigen/Eigen>
#ifdef was_SYCL_DEVICE_ONLY
#define SYCL_DEVICE_ONLY was_SYCL_DEVICE_ONLY
#undef was_SYCL_DEVICE_ONLY
#endif

// OpenImageIO
#include <OpenImageIO/platform.h>
#ifndef __CUDA_ARCH__
// * __CUDACC__ defines whether nvcc is steering compilation or not
// * __CUDA_ARCH__ is always undefined when compiling host code, steered by nvcc or not
// * __CUDA_ARCH__ is only defined for the device code trajectory of compilation steered by nvcc
// If __CUDACC__ then platform.h defines OIIO_HOSTDEVICE as `__host__ __device__`, even if
// __CUDA_ARCH__ is not defined. This might be fine for nvcc, but not clang, which complaints that:
// > error: no function template matches function template specialization 'clamp'
// > note: candidate template ignored: target attributes do not match
#undef OIIO_HOSTDEVICE
#define OIIO_HOSTDEVICE
#endif
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <catch2/catch.hpp>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/Numeric.hpp>
#include <convfelt/iter.hpp>
#include <convfelt/memory.hpp>

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

				[[maybe_unused]] auto const check_output = [&]
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
				sycl::queue q{sycl::gpu_selector_v};
				sycl::range<1> work_items{a.size()};
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());
				sycl::buffer<float> buff_c(c.data(), c.size());

				using Allocator = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
				std::vector<float, Allocator> vals(Allocator{q});
				vals.push_back(1);
				vals.push_back(2);

				q.submit(
					[&](sycl::handler & cgh)
					{
						auto access_a = buff_a.get_access<sycl::access::mode::read>(cgh);
						auto access_b = buff_b.get_access<sycl::access::mode::read>(cgh);
						auto access_c = buff_c.get_access<sycl::access::mode::write>(cgh);

						cgh.parallel_for<class vector_add>(
							work_items,
							[=](sycl::id<1> tid)
							{ access_c[tid] = access_a[tid] + access_b[tid] + vals[0] + vals[1]; });
					});
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {3.f, 7.f, 3.f, 11.f, 3.f};

				CHECK(c == expected);
			}
		}
	}
}

SCENARIO("Basic oneMKL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f};
		std::vector<float> b = {-1.f, 2.f, -3.f, 4.f, -5.f};
		assert(a.size() == b.size());

		WHEN("vectors are added using oneMKL")
		{
			{
				sycl::gpu_selector selector;
				sycl::queue q{selector};
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());

				// NOTE: if a segfault happens here it's because the ERROR_MSG is nullptr, which
				// means there are no enabled backend libraries.
				oneapi::mkl::blas::column_major::axpy(
					q, static_cast<long>(a.size()), 1.0f, buff_a, 1, buff_b, 1);
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {0.f, 4.f, 0.f, 8.f, 0.f};

				CHECK(b == expected);
			}
		}
	}
}

SCENARIO("SyCL with ConvGrid")
{
	GIVEN("Shared grid")
	{
		sycl::context ctx;
		sycl::device dev{sycl::gpu_selector_v};
		using ConvGrid = convfelt::ConvGridTD<float, 3, convfelt::UsmSharedAllocator>;

		auto const pgrid = convfelt::make_unique_sycl<ConvGrid>(
			dev, ctx, Felt::Vec3i{4, 4, 3}, Felt::Vec2i{2, 2}, ctx, dev);

		std::fill(pgrid->data().begin(), pgrid->data().end(), 3);
		CHECK(pgrid->children().data().size() > 1);
		CHECK(pgrid->children().data()[0].data().size() > 1);
		CHECK(&pgrid->children().data()[0].data()[0] == &pgrid->data()[0]);

		WHEN("grid data is doubled using sycl")
		{
			sycl::range<1> work_items{pgrid->children().data().size()};

			sycl::queue q{ctx, dev};
			q.submit(
				[&](sycl::handler & cgh)
				{
				  cgh.prefetch(pgrid->data().data(), pgrid->data().size());
				});
			q.submit(
				[&](sycl::handler & cgh)
				{
					cgh.prefetch(pgrid->children().data().data(), pgrid->children().data().size());
				});
			q.submit(
				[&](sycl::handler & cgh)
				{
					sycl::stream os{2048, 256, cgh};
					[[maybe_unused]] auto const scoped_stream = pgrid->scoped_stream(&os);

					cgh.parallel_for<class grid_mult>(
						work_items,
						[pgrid = pgrid.get()](sycl::id<1> tid)
						{
							for (auto & val : convfelt::iter::val(pgrid->children().get(tid)))
								val *= 2;
						});
				});
			q.wait_and_throw();
			THEN("result is as expected")
			{
				for (auto const val : convfelt::iter::val(*pgrid)) CHECK(val == 6);
			}
		}
	}
}

SCENARIO("Applying filter to ConvGrid")
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

		AND_GIVEN("image is split into filter regions")
		{
			sycl::context ctx;
			sycl::device dev{sycl::gpu_selector_v};
			// sycl::device dev{sycl::cpu_selector_v};
			using FilterGrid =
				convfelt::ConvGridTD<convfelt::Scalar, 3, convfelt::UsmSharedAllocator>;

			const Felt::NodeIdx filter_stride = 2;

			auto filter_input_grid = [&]
			{
				Felt::Vec2i const filter_input_window{4, 4};
				Felt::Vec3i const filter_input_shape{4, 4, 3};

				Felt::Vec3i num_filters = (image_grid.size() - filter_input_shape);
				num_filters = num_filters / filter_stride + Felt::Vec3i::Ones();
				Felt::Vec3i const num_connections =
					(num_filters.array() * filter_input_shape.array()).matrix();

				return convfelt::make_unique_sycl<FilterGrid>(
					dev, ctx, num_connections, filter_input_window, ctx, dev);
			}();
			const Felt::Vec3i filter_input_shape = filter_input_grid->child_size();

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid->children()))
			{
				const Felt::Vec3i input_pos_start =
					filter_input_grid->children().index(filter_pos_idx) * filter_stride;
				(void)input_pos_start;

				for (Felt::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const Felt::Vec3i input_pos =
						input_pos_start + Felt::index<3>(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
				}
			}

			AND_GIVEN("an output grid and weight matrix")
			{
				Felt::Vec2i const filter_output_window{2, 2};
				Felt::Vec3i const filter_output_shape{2, 2, 4};
				Felt::Vec3i const filter_output_grid_size =
					(filter_input_grid->children().size().array() * filter_output_shape.array())
						.matrix();

				auto filter_output_grid = convfelt::make_unique_sycl<FilterGrid>(
					dev, ctx, filter_output_grid_size, filter_output_window, ctx, dev);

				REQUIRE(
					filter_output_grid->children().size() == filter_input_grid->children().size());

				using MatrixMap =
					Eigen::Map<Eigen::Matrix<convfelt::Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0>;
				using ColVectorMap =
					Eigen::Map<Eigen::Matrix<convfelt::Scalar, Eigen::Dynamic, 1>, 0>;

				std::size_t weights_size = static_cast<std::size_t>(filter_output_shape.prod()) *
					static_cast<std::size_t>(filter_input_shape.prod());
				auto weights_data = sycl::malloc_shared<convfelt::Scalar>(weights_size, dev, ctx);

				MatrixMap weights{
					weights_data, filter_output_shape.prod(), filter_input_shape.prod()};
				weights.setRandom();
				//				Eigen::Matrix<convfelt::Scalar, Eigen::Dynamic, 1> wrong{
				//					filter_output_grid->child_size().prod(), 1};
				//				wrong.setConstant(1);

				auto const assert_expected_values = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid->children()))
					{
						const Felt::Vec3i filter_pos =
							filter_input_grid->children().index(filter_pos_idx);
						CAPTURE(filter_pos);

						auto const input_matrix =
							filter_input_grid->children().get(filter_pos_idx).matrix();
						auto const output_matrix =
							filter_output_grid->children().get(filter_pos_idx).matrix();
						CHECK(weights * input_matrix == output_matrix);
					}
				};

				WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")
				{
					sycl::range<1> work_items{filter_input_grid->children().data().size()};
					sycl::queue q{ctx, dev};
					q.submit(
						[&](sycl::handler & cgh)
						{
							sycl::stream os{2048, 256, cgh};
							[[maybe_unused]] auto const scoped_input_stream =
								filter_input_grid->scoped_stream(&os);
							[[maybe_unused]] auto const scoped_output_stream =
								filter_output_grid->scoped_stream(&os);

							cgh.parallel_for<class grid_mult>(
								work_items,
								[filter_input_grid = filter_input_grid.get(),
								 filter_output_grid = filter_output_grid.get(),
								 weights](sycl::id<1> tid)
								{
									auto const & input_child =
										filter_input_grid->children().get(tid);
									auto & output_child = filter_output_grid->children().get(tid);

									ColVectorMap const input_vec{
										input_child.data().data(),
										Eigen::Index(input_child.data().size()),
										1};
									ColVectorMap output_vec{
										output_child.data().data(),
										Eigen::Index(output_child.data().size()),
										1};

									output_vec = weights * input_vec;
								});
						});
					q.wait_and_throw();

					THEN("values are as expected")
					{
						assert_expected_values();
					}

					sycl::free(weights_data, ctx);
				}

				WHEN("filter is applied to grid using oneMKL")
				{
					sycl::queue q{ctx, dev};

					oneapi::mkl::blas::column_major::gemm(
						q,
						oneapi::mkl::transpose::nontrans,
						oneapi::mkl::transpose::nontrans,
						// m: Rows of output / rows of weights
						filter_output_grid->child_size().prod(),
						// n: Columns of output / columns of input
						static_cast<int64_t>(filter_output_grid->children().data().size()),
						// k: Rows of input / columns of weights
						filter_input_grid->child_size().prod(),
						// alpha
						1,
						// a: weights
						weights_data,
						weights.rows(),
						// b: input
						filter_input_grid->data().data(),
						filter_input_grid->child_size().prod(),
						// beta
						0,
						// c: output
						filter_output_grid->data().data(),
						filter_output_grid->child_size().prod());

					q.wait_and_throw();
					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}
			}
		}
	}
}
