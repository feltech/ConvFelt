#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, DontAlignCols, " ", ",", "", "", "(", ")")

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
#include <convfelt/felt2/typedefs.hpp>
#include <convfelt/iter.hpp>
#include <convfelt/memory.hpp>

SCENARIO("Using OpenImageIO with cppcoro and loading into Felt grid")
{
	GIVEN("a simple monochrome image file")
	{
		static constexpr std::string_view file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		using Pixel = felt2::VecDT<float, 3>;

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
				image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

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
				image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

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

felt2::Vec3i input_per_filter_size_from_source_and_filter_size(
	felt2::Vec3i const & source_size,
	felt2::Vec3i const & filter_size,
	felt2::Vec3i const & filter_stride)
{
	felt2::Vec3i input_per_filter_size = felt2::Vec3i::Ones();
	auto const source_window = source_size.head<2>();
	auto const filter_window = filter_size.head<2>();
	auto const stride_window = filter_stride.head<2>();
	auto output_window = input_per_filter_size.head<2>();
	output_window += ((source_window - filter_window).array() / stride_window.array()).matrix();
	return input_per_filter_size;
}

felt2::Vec3i input_size_from_source_and_filter_size(
	felt2::Vec3i const & source_size,
	felt2::Vec3i const & filter_size,
	felt2::Vec3i const & filter_stride)
{
	felt2::Vec3i const input_per_filter_size =
		input_per_filter_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

	return (input_per_filter_size.array() * filter_size.array()).matrix();
}

felt2::Vec3i input_pos_to_source_pos(
	felt2::Vec3i const & filter_size,
	felt2::Vec3i const & filter_stride,
	felt2::Vec3i const & input_pos)
{
	felt2::Vec3i const filter_id = (input_pos.array() / filter_size.array()).matrix();
	auto filter_input_start_pos = (filter_id.array() * filter_size.array()).matrix();
	auto filter_source_start_pos = (filter_stride.array() * filter_id.array()).matrix();
	auto input_filter_local_pos = input_pos - filter_input_start_pos;
	auto source_pos = filter_source_start_pos + input_filter_local_pos;

	return source_pos;
}

template <typename T>
concept IsCallableWithGlobalPos = requires(T t) {
	{
		t(std::declval<felt2::Vec3i>())
	};
};

template <typename T>
concept IsCallableWithFilterPos = requires(T t) {
	{
		t(std::declval<felt2::Vec3i>(), std::declval<felt2::Vec3i>())
	};
};

template <typename T>
concept IsCallableWithPos = IsCallableWithGlobalPos<T> || IsCallableWithFilterPos<T>;

void source_pos_to_input_pos(
	[[maybe_unused]] felt2::Vec3i const & input_size,
	[[maybe_unused]] felt2::Vec3i const & filter_size,
	[[maybe_unused]] felt2::Vec3i const & filter_stride,
	[[maybe_unused]] felt2::Vec3i const & source_size,
	[[maybe_unused]] felt2::Vec3i const & input_per_filter_size,
	[[maybe_unused]] felt2::Vec3i const & source_pos,
	IsCallableWithPos auto && callback)
{
	auto one = felt2::Vec3i::Constant(1);
	auto zero = felt2::Vec3i::Constant(0);

	felt2::Vec3i filter_pos_first = zero;
	felt2::Vec3i filter_pos_last = zero;

	[[maybe_unused]] auto const one_window = one.head<2>();
	auto const zero_window = zero.head<2>();
	auto const source_size_window = source_size.head<2>();
	auto const source_pos_window = source_pos.head<2>();
	auto const filter_size_window = filter_size.head<2>();
	auto const filter_stride_window = filter_stride.head<2>();
	auto filter_pos_first_window = filter_pos_first.head<2>();
	auto filter_pos_last_window = filter_pos_last.head<2>();

	filter_pos_last_window.array() =
		source_pos_window.array().min(source_size_window.array() - filter_size_window.array()) /
		filter_stride_window.array();
	// filter_size - filter_stride = source_pos - filter_pos_first * filter_stride
	// <=> filter_pos_first = (source_pos - filter_size + filter_stride) / filter_stride
	filter_pos_first_window.array() =
		(source_pos_window - filter_size_window + filter_stride_window)
			.array()
			.max(zero_window.array()) /
		filter_stride_window.array();

	for (felt2::Dim x = filter_pos_first(0); x <= filter_pos_last(0); ++x)
		for (felt2::Dim y = filter_pos_first(1); y <= filter_pos_last(1); ++y)
		{
			felt2::Vec3i const filter_pos{x, y, 0};
			auto source_filter_start_pos = (filter_pos.array() * filter_stride.array()).matrix();
			auto filter_source_local_pos = source_pos - source_filter_start_pos;
			auto input_filter_start_pos = (filter_pos.array() * filter_size.array()).matrix();
			auto input_pos = input_filter_start_pos + filter_source_local_pos;

			if constexpr (IsCallableWithGlobalPos<decltype(callback)>)
			{
				callback(input_pos);
			}
			else if constexpr (IsCallableWithFilterPos<decltype(callback)>)
			{
				callback(filter_pos, input_pos);
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
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

		convfelt::InputGrid image_grid{
			{image.spec().height + 2, image.spec().width + 2, image_grid_spec.nchannels},
			{0, 0, 0},
			0};

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.data().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		WHEN("image is split into filter regions")
		{
			using FilterGrid = convfelt::ConvGrid;

			felt2::Vec3i const filter_stride{2, 2, 0};

			FilterGrid filter_input_grid = [&image_grid, &filter_stride]
			{
				felt2::Vec2i const filter_input_window{4, 4};
				felt2::Vec3i const filter_input_shape{4, 4, 3};
				felt2::Vec3i const input_size = input_size_from_source_and_filter_size(
					image_grid.size(), filter_input_shape, filter_stride);

				return FilterGrid{input_size, filter_input_window};
			}();
			const felt2::Vec3i filter_input_shape = filter_input_grid.child_size();

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				const felt2::Vec3i input_pos_start =
					(filter_input_grid.children().index(filter_pos_idx).array() *
					 filter_stride.array())
						.matrix();
				(void)input_pos_start;

				for (felt2::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const felt2::Vec3i input_pos =
						input_pos_start + felt2::index<3>(local_pos_idx, filter.size());

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
							felt2::Vec3i const image_grid_pos =
								input_pos_to_source_pos(filter_input_shape, filter_stride, pos);

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
				felt2::Vec3i const filter_output_shape{2, 2, 4};
				felt2::Vec3i const filter_output_grid_size =
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
						felt2::Scalar const expected =
							filter_input_grid.children().get(filter_pos_idx).matrix().sum();

						const felt2::Vec3i filter_pos =
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
		using ConvGrid = convfelt::ConvGridTD<float, 3, true>;

		auto const pgrid = convfelt::make_unique_sycl<ConvGrid>(
			dev, ctx, felt2::Vec3i{4, 4, 3}, felt2::Vec2i{2, 2}, ctx, dev);

		std::fill(pgrid->data().begin(), pgrid->data().end(), 3);
		CHECK(pgrid->children().data().size() > 1);
		CHECK(pgrid->children().data()[0].data().size() > 1);
		CHECK(&pgrid->children().data()[0].data()[0] == &pgrid->data()[0]);

		WHEN("grid data is doubled using sycl")
		{
			sycl::range<1> work_items{pgrid->children().data().size()};

			sycl::queue q{ctx, dev};
			q.submit([&](sycl::handler & cgh)
					 { cgh.prefetch(pgrid->data().data(), pgrid->data().size()); });
			q.submit(
				[&](sycl::handler & cgh) {
					cgh.prefetch(pgrid->children().data().data(), pgrid->children().data().size());
				});
			q.submit(
				[&](sycl::handler & cgh)
				{
					sycl::stream os{2048, 256, cgh};
					pgrid->set_stream(&os);

					cgh.parallel_for<class grid_mult>(
						work_items,
						[pgrid = pgrid.get()](sycl::id<1> tid)
						{
							for (auto & val : convfelt::iter::val(pgrid->children().get(tid)))
								val *= 2;
						});
				});

			pgrid->set_stream(nullptr);
			// Host-side now, so should have stream no matter what.
			CHECK(pgrid->has_stream());

			pgrid->get_stream() << "Testing host-side streaming works\n";

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
	GIVEN("stride size 3x3, filter size 3x3 and image size 3x3 with 4 channels")
	{
		felt2::Vec3i const filter_stride{3, 3, 0};
		felt2::Vec3i const filter_size{3, 3, 4};

		felt2::Vec3i const source_size{3, 3, 4};

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{3, 3, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{2, 2, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{3, 2, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		WHEN("number of required filters is calculated")
		{
			felt2::Vec3i const size = input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);

			THEN("output size is calculated correctly")
			{
				CHECK(size == felt2::Vec3i{1, 1, 1});
			}
		}
		WHEN("input grid size is calculated")
		{
			felt2::Vec3i const input_size =
				input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

			THEN("input size is calculated correctly")
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

		felt2::Vec3i const input_per_filter_size =
			input_per_filter_size_from_source_and_filter_size(
				source_size, filter_size, filter_stride);
		felt2::Vec3i const input_size =
			input_size_from_source_and_filter_size(source_size, filter_size, filter_stride);

		CHECK(input_size == felt2::Vec3i{6, 4, 4});

		WHEN("input grid minimum point is mapped to source grid")
		{
			felt2::Vec3i const input_pos{0, 0, 0};
			felt2::Vec3i const source_pos =
				input_pos_to_source_pos(filter_size, filter_stride, input_pos);

			THEN("position is as expected")
			{
				CHECK(source_pos == felt2::Vec3i{0, 0, 0});
			}
		}

		WHEN("input grid maximum point is mapped to source grid")
		{
			felt2::Vec3i const input_pos{5, 3, 3};
			felt2::Vec3i const source_pos =
				input_pos_to_source_pos(filter_size, filter_stride, input_pos);

			THEN("position is as expected")
			{
				CHECK(source_pos == felt2::Vec3i{3, 3, 3});
			}
		}

		WHEN("source grid minimum point is mapped to input grid")
		{
			felt2::Vec3i const source_pos{0, 0, 0};

			using PosArray = std::vector<felt2::Vec3i>;
			PosArray input_pos_list;
			source_pos_to_input_pos(
				input_size,
				filter_size,
				filter_stride,
				source_size,
				input_per_filter_size,
				source_pos,
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

			THEN("positions care as expected")
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
			source_pos_to_input_pos(
				input_size,
				filter_size,
				filter_stride,
				source_size,
				input_per_filter_size,
				source_pos,
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

			THEN("position is as expected")
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
			source_pos_to_input_pos(
				input_size,
				filter_size,
				filter_stride,
				source_size,
				input_per_filter_size,
				source_pos,
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

			THEN("position is as expected")
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
			source_pos_to_input_pos(
				input_size,
				filter_size,
				filter_stride,
				source_size,
				input_per_filter_size,
				source_pos,
				[&](const felt2::Vec3i & filter_pos, const felt2::Vec3i & global_pos)
				{
					filter_pos_list.push_back(filter_pos);
					global_pos_list.push_back(global_pos);
				});

			THEN("position is as expected")
			{
				CAPTURE(filter_pos_list);
				CAPTURE(global_pos_list);
				CHECK(std::ranges::equal(filter_pos_list, PosArray{felt2::Vec3i{0, 1, 0}}));
				CHECK(std::ranges::equal(global_pos_list, PosArray{felt2::Vec3i{0, 2, 0}}));
			}
		}
	}

	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{file_path}};
		image.read();
		auto image_grid_spec = image.spec();
		image_grid_spec.width += 2;
		image_grid_spec.height += 2;
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

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
			//			sycl::device dev{sycl::cpu_selector_v};
			using FilterGrid = convfelt::ConvGridTD<felt2::Scalar, 3, true>;

			const felt2::Vec3i filter_stride{2, 2, 0};
			felt2::Vec3i const filter_size{4, 4, 3};

			felt2::Vec3i const input_size = input_size_from_source_and_filter_size(
				image_grid.size(), filter_size, filter_stride);

			auto const input_per_filter_size = input_per_filter_size_from_source_and_filter_size(
				image_grid.size(), filter_size, filter_stride);

			auto filter_input_grid = convfelt::make_unique_sycl<FilterGrid>(
				dev, ctx, input_size, filter_size.head<2>(), ctx, dev);

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid->children()))
			{
				const felt2::Vec3i input_pos_start =
					(filter_input_grid->children().index(filter_pos_idx).array() *
					 filter_stride.array())
						.matrix();

				for (felt2::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const felt2::Vec3i input_pos =
						input_pos_start + felt2::index<3>(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
				}
			}

			WHEN("image is split into filter regions on device")
			{
				auto image_grid_device =
					convfelt::make_unique_sycl<convfelt::ByValue<felt2::Scalar, 3, true>>(
						dev, ctx, image_grid.size(), image_grid.offset(), 0.0f, ctx, dev);

				image_grid_device->data().assign(
					image_grid.data().begin(), image_grid.data().end());

				auto filter_input_grid_device = convfelt::make_unique_sycl<FilterGrid>(
					dev, ctx, input_size, filter_size.head<2>(), ctx, dev);
				sycl::range<1> work_items{image_grid_device->data().size()};
				sycl::queue q{ctx, dev};

				q.submit(
					[&](sycl::handler & cgh)
					{
						sycl::stream os{2048, 512, cgh};
						image_grid_device->set_stream(&os);
						filter_input_grid_device->set_stream(&os);

						cgh.parallel_for<class grid_copy>(
							work_items,
							[input_size,
							 filter_stride,
							 filter_size,
							 input_per_filter_size,
							 image_grid_device = image_grid_device.get(),
							 filter_input_grid_device =
								 filter_input_grid_device.get()](sycl::item<1> item)
							{
								auto & filters = filter_input_grid_device->children();

								[[maybe_unused]] felt2::PosIdx const input_pos_idx =
									item.get_linear_id();
								[[maybe_unused]] felt2::Vec3i const source_pos =
									image_grid_device->index(input_pos_idx);

								felt2::Scalar const source_value =
									image_grid_device->get(input_pos_idx);

								source_pos_to_input_pos(
									input_size,
									filter_size,
									filter_stride,
									image_grid_device->size(),
									input_per_filter_size,
									source_pos,
									[&](felt2::Vec3i const & filter_pos,
										felt2::Vec3i const & global_pos)
									{ filters.get(filter_pos).set(global_pos, source_value); });
							});
					});
				q.wait_and_throw();
				THEN("device grid matches expected grid")
				{
					CHECK(filter_input_grid->size() == filter_input_grid_device->size());
					CHECK(
						filter_input_grid->children().size() ==
						filter_input_grid_device->children().size());
					CHECK(
						filter_input_grid->data().size() ==
						filter_input_grid_device->data().size());
					CHECK(
						filter_input_grid->children().data().size() ==
						filter_input_grid_device->children().data().size());

					for (felt2::PosIdx child_pos_idx :
						 convfelt::iter::pos_idx(filter_input_grid->children()))
					{
						auto const & expected_child =
							filter_input_grid->children().get(child_pos_idx);
						auto const & actual_child =
							filter_input_grid_device->children().get(child_pos_idx);

						CHECK(expected_child.size() == actual_child.size());
						CHECK(expected_child.offset() == actual_child.offset());
						CHECK(expected_child.data().size() == actual_child.data().size());

						CHECK(std::ranges::equal(expected_child.data(), actual_child.data()));
						//						for (felt2::PosIdx leaf_pos_idx :
						// convfelt::iter::pos_idx(actual_child))
						//						{
						//							felt2::Scalar const expected_value =
						// expected_child.get(leaf_pos_idx); felt2::Scalar const actual_value =
						// actual_child.get(leaf_pos_idx); INFO(actual_child.index(leaf_pos_idx));
						// CHECK(expected_value
						// ==
						// actual_value);
						//						}
					}
				}
			}

			AND_GIVEN("an output grid and weight matrix")
			{
				felt2::Vec2i const filter_output_window{2, 2};
				felt2::Vec3i const filter_output_shape{2, 2, 4};
				felt2::Vec3i const filter_output_grid_size =
					(filter_input_grid->children().size().array() * filter_output_shape.array())
						.matrix();

				auto filter_output_grid = convfelt::make_unique_sycl<FilterGrid>(
					dev, ctx, filter_output_grid_size, filter_output_window, ctx, dev);

				REQUIRE(
					filter_output_grid->children().size() == filter_input_grid->children().size());

				using MatrixMap =
					Eigen::Map<Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0>;
				using ColVectorMap = Eigen::Map<Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, 1>, 0>;

				std::size_t weights_size = static_cast<std::size_t>(filter_output_shape.prod()) *
					static_cast<std::size_t>(filter_size.prod());
				auto weights_data = sycl::malloc_shared<felt2::Scalar>(weights_size, dev, ctx);

				MatrixMap weights{weights_data, filter_output_shape.prod(), filter_size.prod()};
				weights.setRandom();
				//				Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, 1> wrong{
				//					filter_output_grid->child_size().prod(), 1};
				//				wrong.setConstant(1);

				auto const assert_expected_values = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid->children()))
					{
						const felt2::Vec3i filter_pos =
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
					sycl::nd_range<1> work_range{
						filter_output_grid->data().size(),
						static_cast<size_t>(filter_output_grid->child_size().prod())};
					sycl::queue q{ctx, dev};

					q.submit(
						[&](sycl::handler & cgh)
						{
							sycl::stream os{2048, 256, cgh};
							filter_input_grid->set_stream(&os);
							filter_output_grid->set_stream(&os);

							assert(weights.rows() == filter_output_grid->child_size().prod());
							assert(weights.cols() == filter_input_grid->child_size().prod());

							sycl::accessor<
								felt2::Scalar,
								1,
								sycl::access::mode::read_write,
								sycl::access::target::local>
								input_child_data(
									sycl::range<1>(static_cast<size_t>(weights.cols())), cgh);

							cgh.parallel_for<class grid_mult>(
								work_range,
								[filter_input_grid = filter_input_grid.get(),
								 filter_output_grid = filter_output_grid.get(),
								 weights,
								 input_child_data](sycl::nd_item<1> item)
								{
									std::size_t const group_id = item.get_group_linear_id();
									std::size_t const local_id = item.get_local_linear_id();

									auto & input_child =
										filter_input_grid->children().get(group_id);
									auto & output_child =
										filter_output_grid->children().get(group_id);

									assert(input_child.data().size() == input_child_data.size());
									assert(local_id < output_child.data().size());

									item.async_work_group_copy(
											input_child_data.get_pointer(),
											sycl::global_ptr<felt2::Scalar>{
												input_child.data().data()},
											input_child.data().size())
										.wait();

									ColVectorMap const input_vec{
										input_child_data.get_pointer().get(),
										Eigen::Index(input_child.data().size()),
										1};
									ColVectorMap output_vec{
										output_child.data().data(),
										Eigen::Index(output_child.data().size()),
										1};

									auto const row_idx = static_cast<Eigen::Index>(local_id);
									output_vec(row_idx) = weights.row(row_idx).dot(input_vec);
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
						filter_output_grid->matrix().rows(),
						// n: Columns of output / columns of input
						filter_output_grid->matrix().cols(),
						// k: Rows of input / columns of weights
						filter_input_grid->matrix().rows(),
						// alpha
						1,
						// a: weights
						weights_data,
						weights.rows(),
						// b: input
						filter_input_grid->matrix().data(),
						filter_input_grid->matrix().rows(),
						// beta
						0,
						// c: output
						filter_output_grid->matrix().data(),
						filter_output_grid->matrix().rows());

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
