#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, DontAlignCols, " ", ",", "", "", "(", ")")

#include <ranges>
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

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
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

				OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
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

template <felt2::Dim D = 3>
struct FilterSizeHelper
{
	using VecDi = felt2::VecDi<D>;

	VecDi const filter_size;
	VecDi const filter_stride{[]() constexpr
							  {
								  VecDi default_stride;
								  default_stride.template head<D - 1>().setConstant(1);
								  default_stride.template tail<1>().setConstant(0);
								  return default_stride;
							  }()};

	[[nodiscard]] constexpr auto num_filter_elems() const noexcept
	{
		return filter_size.prod();
	}

	[[nodiscard]] felt2::VecDi<D - 1> filter_window() const
	{
		return filter_size.template head<D - 1>();
	}

	[[nodiscard]] VecDi input_size_from_source_size(VecDi const & source_size) const
	{
		return input_size_from_source_and_filter_size(filter_size, filter_stride, source_size);
	}

	[[nodiscard]] VecDi input_start_pos_from_filter_pos(VecDi const & filter_pos) const
	{
		return (filter_pos.array() * filter_stride.array()).matrix();
	}

	[[nodiscard]] VecDi source_pos_from_input_pos(VecDi const & input_pos) const
	{
		return source_pos_from_input_pos_and_filter_size(filter_size, filter_stride, input_pos);
	}

	[[nodiscard]] VecDi output_size_from_num_filter_regions(VecDi const & num_filter_regions) const
	{
		return (num_filter_regions.array() * filter_size.array()).matrix();
	}

	void input_pos_from_source_pos(
		VecDi const & source_size,
		VecDi const & source_pos,
		IsCallableWithPos auto && callback) const
	{
		input_pos_from_source_pos_and_filter_size(
			filter_size,
			filter_stride,
			source_size,
			source_pos,
			std::forward<decltype(callback)>(callback));
	}

	/**
	 * Assuming we wish to construct a grid storing all filter inputs side-by-side, given a source
	 * image size calculate how many distinct regions will need to be stacked side-by-side.
	 *
	 * @param filter_size Size of filter to walk across source image.
	 * @param filter_stride Size of each step as the filter walks across the source image.
	 * @param source_size Size of source image.
	 * @return Number of regions (along each dimension) stamped out by the filter after it has
	 * walked the source image.
	 */
	static VecDi num_filter_regions_from_source_and_filter_size(
		VecDi const & filter_size, VecDi const & filter_stride, VecDi const & source_size)
	{
		VecDi num_filter_regions = VecDi::Ones();
		auto const source_window = source_size.template head<2>();
		auto const filter_window = filter_size.template head<2>();
		auto const stride_window = filter_stride.template head<2>();
		auto output_window = num_filter_regions.template head<2>();
		output_window += ((source_window - filter_window).array() / stride_window.array()).matrix();
		return num_filter_regions;
	}

	/**
	 * Assuming we wish to construct a grid storing all filter inputs side-by-side, given a source
	 * image size calculate how large the grid of filter inputs must be.
	 *
	 * @param filter_size Size of filter to walk across source image.
	 * @param filter_stride Size of each step as the filter walks across the source image.
	 * @param source_size Size of source image.
	 * @return Required size to store all filter inputs side-by-side.
	 */
	static VecDi input_size_from_source_and_filter_size(
		VecDi const & filter_size, VecDi const & filter_stride, VecDi const & source_size)
	{
		VecDi const input_per_filter_size =
			num_filter_regions_from_source_and_filter_size(filter_size, filter_stride, source_size);

		return (input_per_filter_size.array() * filter_size.array()).matrix();
	}

	/**
	 * Assuming a grid of all filter inputs side-by-side, given a position in this grid, get the
	 * corresponding position in the source image that the filter input element was copied from.
	 *
	 * @param filter_size Size of filter to walk across source image.
	 * @param filter_stride Size of each step as the filter walks across the source image.
	 * @param input_pos Position in grid of all filter inputs side-by-side.
	 * @return
	 */
	static VecDi source_pos_from_input_pos_and_filter_size(
		VecDi const & filter_size, VecDi const & filter_stride, VecDi const & input_pos)
	{
		VecDi const filter_id = (input_pos.array() / filter_size.array()).matrix();
		auto filter_input_start_pos = (filter_id.array() * filter_size.array()).matrix();
		auto filter_source_start_pos = (filter_id.array() * filter_stride.array()).matrix();
		auto filter_local_pos = input_pos - filter_input_start_pos;
		auto source_pos = filter_source_start_pos + filter_local_pos;

		return source_pos;
	}

	/**
	 * Calculate the position(s) in a grid of all filter inputs side-by-side that correspond to a
	 * given position in the source image.
	 *
	 * That is, map a source image pixel to the (multiple) filter input image pixel(s).
	 *
	 * @param filter_size Size of filter to walk across source image.
	 * @param filter_stride Size of each step as the filter walks across the source image.
	 * @param source_size Size of source image.
	 * @param source_pos Position within source image.
	 * @param callback Callback to call, passing the positions in the filter input image that
	 * correspond to the @p source_pos position in the source image..
	 */
	static void input_pos_from_source_pos_and_filter_size(
		VecDi const & filter_size,
		VecDi const & filter_stride,
		VecDi const & source_size,
		VecDi const & source_pos,
		IsCallableWithPos auto && callback)
	{
		auto one = VecDi::Constant(1);
		auto zero = VecDi::Constant(0);

		VecDi filter_pos_first = zero;
		VecDi filter_pos_last = zero;

		[[maybe_unused]] auto const one_window = one.template head<D - 1>();
		auto const zero_window = zero.template head<D - 1>();
		auto const source_size_window = source_size.template head<D - 1>();
		auto const source_pos_window = source_pos.template head<D - 1>();
		auto const filter_size_window = filter_size.template head<D - 1>();
		auto const filter_stride_window = filter_stride.template head<D - 1>();
		auto filter_pos_first_window = filter_pos_first.template head<D - 1>();
		auto filter_pos_last_window = filter_pos_last.template head<D - 1>();

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

		// TODO(DF): Make n-dimensional
		for (felt2::Dim x = filter_pos_first(0); x <= filter_pos_last(0); ++x)
			for (felt2::Dim y = filter_pos_first(1); y <= filter_pos_last(1); ++y)
			{
				VecDi const filter_pos{x, y, 0};
				auto source_filter_start_pos =
					(filter_pos.array() * filter_stride.array()).matrix();
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
};

template <felt2::Dim D, class... Others>
FilterSizeHelper(felt2::VecDi<D>, Others...) -> FilterSizeHelper<D>;

// template <class... Others>
// FilterSizeHelper(std::initializer_list<felt2::PosIdx> size, Others...) ->
// FilterSizeHelper<size.size()>;

SCENARIO("Transforming source image points to filter input grid points and vice versa")
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
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

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
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

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
				[&](const felt2::Vec3i & pos) { input_pos_list.push_back(pos); });

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
				[&](const felt2::Vec3i & filter_pos, const felt2::Vec3i & global_pos)
				{
					filter_pos_list.push_back(filter_pos);
					global_pos_list.push_back(global_pos);
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

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		WHEN("image is split into filter regions")
		{
			FilterSizeHelper filter_input_sizer{felt2::Vec3i{4, 4, 3}, felt2::Vec3i{2, 2, 0}};

			convfelt::ConvGrid filter_input_grid{
				filter_input_sizer.input_size_from_source_size(image_grid.size()),
				filter_input_sizer.filter_size};

			CHECK(filter_input_grid.child_size() == filter_input_sizer.filter_size);

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				const felt2::Vec3i input_pos_start =
					filter_input_sizer.input_start_pos_from_filter_pos(
						filter_input_grid.children().index(filter_pos_idx));

				for (felt2::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const felt2::Vec3i input_pos =
						input_pos_start + felt2::index(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
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
							CHECK(filter_input.get(pos_idx) == image_grid.get(source_pos));

							if (filter_input.get(pos) != 0)
								++num_nonzero;
						}
					}
					CHECK(num_nonzero > 0);
				}
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

		std::fill(pgrid->storage().begin(), pgrid->storage().end(), 3);
		CHECK(pgrid->children().storage().size() > 1);
		CHECK(pgrid->children().storage()[0].storage().size() > 1);
		CHECK(&pgrid->children().storage()[0].storage()[0] == &pgrid->storage()[0]);

		WHEN("grid data is doubled using sycl")
		{
			sycl::range<1> work_items{pgrid->children().storage().size()};

			sycl::queue q{ctx, dev};
			q.submit([&](sycl::handler & cgh)
					 { cgh.prefetch(pgrid->storage().data(), pgrid->storage().size()); });
			q.submit(
				[&](sycl::handler & cgh) {
					cgh.prefetch(
						pgrid->children().storage().data(), pgrid->children().storage().size());
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

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		AND_GIVEN("image is split into filter regions")
		{
			sycl::context ctx;
			sycl::device dev{sycl::gpu_selector_v};
			//			sycl::device dev{sycl::cpu_selector_v};
			using FilterGrid = convfelt::ConvGridTD<felt2::Scalar, 3, true>;

			FilterSizeHelper const filter_input_sizer{felt2::Vec3i{4, 4, 3}, felt2::Vec3i{2, 2, 0}};

			auto input_size = filter_input_sizer.input_size_from_source_size(image_grid.size());

			auto filter_input_grid = convfelt::make_unique_sycl<FilterGrid>(
				dev, ctx, input_size, filter_input_sizer.filter_window(), ctx, dev);

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid->children()))
			{
				const felt2::Vec3i input_pos_start =
					filter_input_sizer.input_start_pos_from_filter_pos(
						filter_input_grid->children().index(filter_pos_idx));

				for (felt2::PosIdx local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const felt2::Vec3i input_pos =
						input_pos_start + felt2::index(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
				}
			}

			WHEN("image is split into filter regions on device")
			{
				auto image_grid_device =
					convfelt::make_unique_sycl<convfelt::ByValue<felt2::Scalar, 3, true>>(
						dev, ctx, image_grid.size(), image_grid.offset(), 0.0f, ctx, dev);

				image_grid_device->storage().assign(
					image_grid.storage().begin(), image_grid.storage().end());

				auto filter_input_grid_device = convfelt::make_unique_sycl<FilterGrid>(
					dev, ctx, input_size, filter_input_sizer.filter_window(), ctx, dev);
				sycl::range<1> work_items{image_grid_device->storage().size()};
				sycl::queue q{ctx, dev};

				q.submit(
					[&](sycl::handler & cgh)
					{
						sycl::stream os{2048, 512, cgh};
						image_grid_device->set_stream(&os);
						filter_input_grid_device->set_stream(&os);

						cgh.parallel_for<class grid_copy>(
							work_items,
							[filter_input_sizer,
							 image_grid_device = image_grid_device.get(),
							 filter_input_grid_device =
								 filter_input_grid_device.get()](sycl::item<1> item)
							{
								auto & filter_inputs = filter_input_grid_device->children();

								felt2::PosIdx const input_pos_idx = item.get_linear_id();
								felt2::Vec3i const source_pos =
									image_grid_device->index(input_pos_idx);

								felt2::Scalar const source_value =
									image_grid_device->get(input_pos_idx);

								filter_input_sizer.input_pos_from_source_pos(
									image_grid_device->size(),
									source_pos,
									[&](felt2::Vec3i const & filter_pos,
										felt2::Vec3i const & global_pos) {
										filter_inputs.get(filter_pos).set(global_pos, source_value);
									});
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
						filter_input_grid->storage().size() ==
						filter_input_grid_device->storage().size());
					CHECK(
						filter_input_grid->children().storage().size() ==
						filter_input_grid_device->children().storage().size());

					for (felt2::PosIdx child_pos_idx :
						 convfelt::iter::pos_idx(filter_input_grid->children()))
					{
						auto const & expected_child =
							filter_input_grid->children().get(child_pos_idx);
						auto const & actual_child =
							filter_input_grid_device->children().get(child_pos_idx);

						CHECK(expected_child.size() == actual_child.size());
						CHECK(expected_child.offset() == actual_child.offset());
						CHECK(expected_child.storage().size() == actual_child.storage().size());

						CHECK(std::ranges::equal(expected_child.storage(), actual_child.storage()));
					}
				}
			}

			AND_GIVEN("an output grid and weight matrix")
			{
				FilterSizeHelper const filter_output_sizer{felt2::Vec3i{2, 2, 4}};

				auto filter_output_grid = convfelt::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					filter_output_sizer.output_size_from_num_filter_regions(
						filter_input_grid->children().size()),
					filter_output_sizer.filter_window(),
					ctx,
					dev);

				REQUIRE(
					filter_output_grid->children().size() == filter_input_grid->children().size());

				convfelt::USMMatrix usm_weights{
					filter_output_sizer.num_filter_elems(),
					filter_input_sizer.num_filter_elems(),
					dev,
					ctx};

				usm_weights.matrix().setRandom();
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
						CHECK(usm_weights.matrix() * input_matrix == output_matrix);
					}
				};

				WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")
				{
					sycl::queue q{ctx, dev};

					q.submit([pgrid = filter_input_grid.get()](sycl::handler & cgh)
							 { cgh.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					q.submit([pgrid = filter_output_grid.get()](sycl::handler & cgh)
							 { cgh.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					q.submit(
						[&usm_weights](sycl::handler & cgh)
						{ cgh.prefetch(usm_weights.bytes().data(), usm_weights.bytes().size()); });

					sycl::nd_range<1> work_range{
						filter_output_grid->storage().size(),
						static_cast<size_t>(filter_output_grid->child_size().prod())};

					q.submit(
						[&](sycl::handler & cgh)
						{
							sycl::stream os{2048, 256, cgh};
							filter_input_grid->set_stream(&os);
							filter_output_grid->set_stream(&os);

							assert(
								usm_weights.matrix().rows() ==
								filter_output_grid->child_size().prod());
							assert(
								usm_weights.matrix().cols() ==
								filter_input_grid->child_size().prod());

							sycl::accessor<
								felt2::Scalar,
								1,
								sycl::access::mode::read_write,
								sycl::access::target::local>
								input_child_data{
									static_cast<std::size_t>(usm_weights.matrix().cols()), cgh};

							cgh.parallel_for<class grid_mult>(
								work_range,
								[filter_input_grid = filter_input_grid.get(),
								 filter_output_grid = filter_output_grid.get(),
								 weights = usm_weights.matrix(),
								 input_child_data](sycl::nd_item<1> item)
								{
									std::size_t const group_id = item.get_group_linear_id();
									std::size_t const local_id = item.get_local_linear_id();

									auto & input_child =
										filter_input_grid->children().get(group_id);
									auto & output_child =
										filter_output_grid->children().get(group_id);

									assert(input_child.storage().size() == input_child_data.size());
									assert(local_id < output_child.storage().size());

									// TODO(DF): Do we need to construct filter input regions in
									//  global memory only to copy out again into local memory? I.e.
									//  will we need the input grid further down the line? Otherwise
									//  could construct input region here - see next test.
									item.async_work_group_copy(
											input_child_data.get_pointer(),
											sycl::global_ptr<felt2::Scalar>{
												input_child.storage().data()},
											input_child.storage().size())
										.wait();

									using ColVectorMap = Eigen::Map<
										Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, 1>,
										Eigen::ColMajor>;

									ColVectorMap const input_vec{
										input_child_data.get_pointer().get(), weights.cols()};
									ColVectorMap output_vec{
										output_child.storage().data(), weights.rows()};

									auto const row_idx = static_cast<Eigen::Index>(local_id);
									output_vec(row_idx) = weights.row(row_idx).dot(input_vec);
								});
						});
					q.wait_and_throw();

					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}  // WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")

				WHEN("filter is applied in kernel that computes filter input on the fly")
				{
					sycl::queue q{ctx, dev};

					auto filter_input_template = convfelt::make_unique_sycl<
						convfelt::TemplateParentGridTD<felt2::Scalar, 3, true>>(
						q.get_device(),
						q.get_context(),
						input_size,
						filter_input_sizer.filter_window(),
						q.get_context(),
						q.get_device());

					auto image_grid_device =
						convfelt::make_unique_sycl<convfelt::ByValue<felt2::Scalar, 3, true>>(
							q.get_device(),
							q.get_context(),
							image_grid.size(),
							image_grid.offset(),
							0.0f,
							q.get_context(),
							q.get_device());

					image_grid_device->storage().assign(
						image_grid.storage().begin(), image_grid.storage().end());

					auto const filters_per_image =
						static_cast<std::size_t>(filter_output_grid->children().size().prod());
					auto const elems_per_filter =
						static_cast<std::size_t>(filter_output_grid->child_size().prod());

					std::size_t const ideal_elems_per_work_group =
						dev.get_info<sycl::info::device::sub_group_sizes>()[0];
					std::size_t const elems_per_image = elems_per_filter * filters_per_image;

					CHECK(
						elems_per_image ==
						static_cast<std::size_t>(filter_output_grid->size().prod()));
					CHECK(elems_per_image == filter_output_grid->storage().size());

					constexpr auto scalar = [](auto v) { return static_cast<felt2::Scalar>(v); };

					felt2::Scalar const ideal_work_groups_per_image =
						scalar(elems_per_image) / scalar(ideal_elems_per_work_group);

					felt2::Scalar const ideal_filters_per_work_group =
						scalar(filters_per_image) / ideal_work_groups_per_image;

					auto const filters_per_work_group =
						static_cast<std::size_t>(std::ceil(ideal_filters_per_work_group));

					felt2::Scalar const work_groups_per_image =
						scalar(filters_per_image) / scalar(filters_per_work_group);

					felt2::Scalar const elems_per_work_group =
						scalar(elems_per_image) / work_groups_per_image;

					auto const num_work_groups =
						static_cast<std::size_t>(std::ceil(work_groups_per_image));
					auto const work_group_size = std::min(
						static_cast<std::size_t>(std::ceil(elems_per_work_group)), elems_per_image);

					CHECK(
						dev.get_info<sycl::info::device::local_mem_size>() > sizeof(felt2::Scalar) *
							static_cast<std::size_t>(filter_input_sizer.num_filter_elems()));
					CAPTURE(dev.get_info<sycl::info::device::sub_group_sizes>());
					CHECK(
						dev.get_info<sycl::info::device::sub_group_sizes>()[0] <= work_group_size);

					// Check invariants
					CHECK(usm_weights.matrix().rows() == filter_output_grid->child_size().prod());
					CHECK(
						usm_weights.matrix().rows() <= static_cast<Eigen::Index>(work_group_size));
					CHECK(usm_weights.matrix().cols() == filter_input_grid->child_size().prod());
					CHECK(usm_weights.matrix().cols() == filter_input_sizer.num_filter_elems());
					// TODO(DF): Technically this check could fail and we still have a valid
					//  situation, just complicated slightly by data boundary condition... or could
					//  it?
					CHECK(
						filter_output_grid->storage().size() == num_work_groups * work_group_size);

					sycl::nd_range<1> work_range{
						num_work_groups * work_group_size, work_group_size};

					q.submit(
						[&](sycl::handler & cgh)
						{
							sycl::stream os{2048, 256, cgh};
							filter_input_grid->set_stream(&os);
							filter_output_grid->set_stream(&os);
							image_grid_device->set_stream(&os);

							sycl::local_accessor<felt2::Scalar, 2> input_child_data{
								sycl::range(
									filters_per_work_group,
									static_cast<std::size_t>(usm_weights.matrix().cols())),
								cgh};

							cgh.parallel_for<class dynamic_input>(
								work_range,
								[image_grid = image_grid_device.get(),
								 filter_input_template = filter_input_template.get(),
								 filter_output_grid = filter_output_grid.get(),
								 filters_per_work_group,
								 weights = usm_weights.matrix(),
								 input_child_data,
								 filter_input_sizer,
								 input_size = filter_input_sizer.input_size_from_source_size(
									 image_grid.size())](sycl::nd_item<1> item)
								{
									std::size_t const group_id = item.get_group_linear_id();
									std::size_t const local_id = item.get_local_linear_id();

									felt2::PosIdx const elems_per_work_group =
										item.get_local_range(0);
									felt2::PosIdx const elems_per_filter =
										elems_per_work_group / filters_per_work_group;

									felt2::PosIdx const filter_idx_offset_for_item =
										local_id / elems_per_filter;
									felt2::PosIdx const filter_idx_for_item =
										group_id * filters_per_work_group +
										filter_idx_offset_for_item;
									felt2::PosIdx const elem_idx_for_item =
										local_id - filter_idx_offset_for_item * elems_per_filter;

									auto const num_cols =
										static_cast<felt2::PosIdx>(weights.cols());

									// Deliberately copy into private memory.
									auto input_child =
										filter_input_template->children().get(filter_idx_for_item);
									input_child.storage() = std::span(
										&input_child_data[filter_idx_offset_for_item][0], num_cols);

									auto & output_child =
										filter_output_grid->children().get(filter_idx_for_item);

									// Assert invariants.
									assert(num_cols == input_child.storage().size());
									assert(
										static_cast<felt2::PosIdx>(weights.rows()) <=
										elems_per_filter);

									for (felt2::PosIdx simd_col = 0; simd_col < num_cols;
										 simd_col += elems_per_filter)
									{
										felt2::PosIdx const col = simd_col + elem_idx_for_item;
										if (col >= num_cols)
											break;

										felt2::Vec3i const pos_in_input_grid =
											input_child.index(col);
										felt2::Vec3i const pos_in_source_grid =
											filter_input_sizer.source_pos_from_input_pos(
												pos_in_input_grid);
										input_child.set(col, image_grid->get(pos_in_source_grid));
									}

									if (elem_idx_for_item > output_child.storage().size())
										return;

									item.barrier(sycl::access::fence_space::local_space);

									auto const row_idx =
										static_cast<Eigen::Index>(elem_idx_for_item);

									output_child.matrix()(row_idx) =
										weights.row(row_idx).dot(input_child.matrix());
								});
						});
					q.wait_and_throw();

					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}  // WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")

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
						usm_weights.matrix().data(),
						usm_weights.matrix().rows(),
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
