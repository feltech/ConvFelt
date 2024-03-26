#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, DontAlignCols, " ", ",", "", "", "(", ")")

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iterator>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <OpenImageIO/typedesc.h>
#include <catch2/catch_message.hpp>
#include <etl/string.h>
#include <experimental/mdspan>  // NOLINT(misc-include-cleaner)
#include <fmt/format.h>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/types.hpp>
#include <sycl/sycl.hpp>

#include "convfelt/felt2/components/sycl.hpp"
#include "convfelt/felt2/index.hpp"

namespace stdex = std::experimental;

// Eigen 3.4.0 (fixed in master) has a workaround for SYCL where it wraps min/max functions in a
// lambda on SYCL. This appears to fix cudaErrorIllegalAddress in min/max functions - something to
// do with function pointers.
// https://gitlab.com/libeigen/eigen/-/commit/e24a1f57e35f3f3894a5612bb8b4e34bf68ebb26
// #define SYCL_DEVICE_ONLY
// #include <Eigen/src/Core/GenericPacketMath.h>
// #undef SYCL_DEVICE_ONLY

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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/felt2/typedefs.hpp>
#include <convfelt/iter.hpp>

// NOLINTBEGIN(readability-function-cognitive-complexity)
// NOLINTBEGIN(*-magic-numbers)
// NOLINTBEGIN(readability-identifier-length)

SCENARIO("Using OpenImageIO with cppcoro and loading into Felt grid")
{
	GIVEN("a simple monochrome image file")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		using Pixel = felt2::VecDT<float, 3>;

		WHEN("image file is loaded")
		{
			auto task = []() -> cppcoro::task<OIIO::ImageBuf>
			{
				OIIO::ImageBuf image_buf{std::string{k_file_path}};
				image_buf.read();
				co_return image_buf;
			};

			OIIO::ImageBuf const image = cppcoro::sync_wait(task());

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
					convfelt::make_host_context(),
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
					convfelt::make_host_context(),
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
concept IsCallableWithGlobalPos = requires(T obj_)
{
	{obj_(std::declval<felt2::Vec3i>())};
};

template <typename T>
concept IsCallableWithFilterPos = requires(T obj_)
{
	{obj_(std::declval<felt2::Vec3i>(), std::declval<felt2::Vec3i>())};
};

template <typename T>
concept IsCallableWithPos = IsCallableWithGlobalPos<T> || IsCallableWithFilterPos<T>;

template <felt2::Dim D = 3>
struct FilterSizeHelper
{
	using VecDi = felt2::VecDi<D>;

	VecDi filter_size;
	VecDi filter_stride{[]() constexpr
						{
							// False positive:
							// NOLINTNEXTLINE(misc-const-correctness)
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

	[[nodiscard]] VecDi input_size_from_source_size(VecDi const & source_size_) const
	{
		return input_size_from_source_and_filter_size(filter_size, filter_stride, source_size_);
	}

	[[nodiscard]] VecDi input_start_pos_from_filter_pos(VecDi const & filter_pos_) const
	{
		return (filter_pos_.array() * filter_stride.array()).matrix();
	}

	[[nodiscard]] VecDi source_pos_from_input_pos(VecDi const & input_pos_) const
	{
		return source_pos_from_input_pos_and_filter_size(filter_size, filter_stride, input_pos_);
	}

	[[nodiscard]] VecDi output_size_from_num_filter_regions(VecDi const & num_filter_regions_) const
	{
		return (num_filter_regions_.array() * filter_size.array()).matrix();
	}

	void input_pos_from_source_pos(
		VecDi const & source_size_,
		VecDi const & source_pos_,
		IsCallableWithPos auto && callback_) const
	{
		input_pos_from_source_pos_and_filter_size(
			filter_size,
			filter_stride,
			source_size_,
			source_pos_,
			std::forward<decltype(callback_)>(callback_));
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
		VecDi const & filter_size_, VecDi const & filter_stride_, VecDi const & source_size_)
	{
		VecDi num_filter_regions = VecDi::Ones();
		auto const source_window = source_size_.template head<2>();
		auto const filter_window = filter_size_.template head<2>();
		auto const stride_window = filter_stride_.template head<2>();
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
		VecDi const & filter_size_, VecDi const & filter_stride_, VecDi const & source_size_)
	{
		VecDi const input_per_filter_size = num_filter_regions_from_source_and_filter_size(
			filter_size_, filter_stride_, source_size_);

		return (input_per_filter_size.array() * filter_size_.array()).matrix();
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
		VecDi const & filter_size_, VecDi const & filter_stride_, VecDi const & input_pos_)
	{
		VecDi const filter_id = (input_pos_.array() / filter_size_.array()).matrix();
		auto filter_input_start_pos = (filter_id.array() * filter_size_.array()).matrix();
		auto filter_source_start_pos = (filter_id.array() * filter_stride_.array()).matrix();
		auto filter_local_pos = input_pos_ - filter_input_start_pos;
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
		VecDi const & filter_size_,
		VecDi const & filter_stride_,
		VecDi const & source_size_,
		VecDi const & source_pos_,
		IsCallableWithPos auto && callback_)
	{
		auto one = VecDi::Constant(1);
		auto zero = VecDi::Constant(0);

		VecDi filter_pos_first = zero;
		VecDi filter_pos_last = zero;

		[[maybe_unused]] auto const one_window = one.template head<D - 1>();
		auto const zero_window = zero.template head<D - 1>();
		auto const source_size_window = source_size_.template head<D - 1>();
		auto const source_pos_window = source_pos_.template head<D - 1>();
		auto const filter_size_window = filter_size_.template head<D - 1>();
		auto const filter_stride_window = filter_stride_.template head<D - 1>();
		auto filter_pos_first_window = filter_pos_first.template head<D - 1>();
		auto filter_pos_last_window = filter_pos_last.template head<D - 1>();

		// Note: causes cudaErrorIllegalAddress in Eigen <= 3.4 - fixed in master.
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
					(filter_pos.array() * filter_stride_.array()).matrix();
				auto filter_source_local_pos = source_pos_ - source_filter_start_pos;
				auto input_filter_start_pos = (filter_pos.array() * filter_size_.array()).matrix();
				auto input_pos = input_filter_start_pos + filter_source_local_pos;

				if constexpr (IsCallableWithGlobalPos<decltype(callback_)>)
				{
					callback_(input_pos);
				}
				else if constexpr (IsCallableWithFilterPos<decltype(callback_)>)
				{
					callback_(filter_pos, input_pos);
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
				[&](const felt2::Vec3i & pos_) { input_pos_list.push_back(pos_); });

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
				[&](const felt2::Vec3i & pos_) { input_pos_list.push_back(pos_); });

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
				[&](const felt2::Vec3i & pos_) { input_pos_list.push_back(pos_); });

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
				[&](const felt2::Vec3i & filter_pos_, const felt2::Vec3i & global_pos_)
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

SCENARIO("Input/output ConvGrids")
{
	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{k_file_path}};
		image.read();
		auto image_grid_spec = image.spec();
		image_grid_spec.width += 2;
		image_grid_spec.height += 2;
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

		convfelt::InputGrid image_grid{
			convfelt::make_host_context(),
			{image.spec().height + 2, image.spec().width + 2, image_grid_spec.nchannels},
			{0, 0, 0},
			0};

		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		WHEN("image is split into filter regions")
		{
			FilterSizeHelper const filter_input_sizer{felt2::Vec3i{4, 4, 3}, felt2::Vec3i{2, 2, 0}};

			convfelt::ConvGrid filter_input_grid{
				convfelt::make_host_context(),
				filter_input_sizer.input_size_from_source_size(image_grid.size()),
				filter_input_sizer.filter_size};

			CHECK(filter_input_grid.child_size() == filter_input_sizer.filter_size);

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid.children()))
			{
				const felt2::Vec3i input_pos_start =
					filter_input_sizer.input_start_pos_from_filter_pos(
						filter_input_grid.children().index(filter_pos_idx));

				for (felt2::PosIdx const local_pos_idx : convfelt::iter::pos_idx(filter))
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

void async_handler(sycl::exception_list error_list_)
{
	if (!error_list_.empty())
	{
		std::string errors;
		for (std::size_t idx = 0; idx < error_list_.size(); ++idx)
		{
			try
			{
				if (error_list_[idx])
				{
					std::rethrow_exception(error_list_[idx]);
				}
			}
			catch (std::exception & e)
			{
				std::format_to(std::back_inserter(errors), "{}: {}\n", idx, e.what());
			}
			catch (...)
			{
				std::format_to(
					std::back_inserter(errors), "{}: <unknown non-std::exception>\n", idx);
			}
		}
		throw std::runtime_error{errors};
	}
}

SCENARIO("Basic SyCL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.F, 2.F, 3.F, 4.F, 5.F};
		std::vector<float> b = {-1.F, 2.F, -3.F, 4.F, -5.F};
		std::vector<float> c(a.size());
		assert(a.size() == b.size());
		sycl::queue queue{sycl::gpu_selector_v, &async_handler};  // NOLINT(misc-const-correctness)

		WHEN("vectors are added using sycl")
		{
			using Allocator = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
			std::vector<float, Allocator> vals(Allocator{queue});
			{
				sycl::range<1> work_items{a.size()};
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());
				sycl::buffer<float> buff_c(c.data(), c.size());

				vals.push_back(1);
				vals.push_back(2);

				queue.submit(
					[&](sycl::handler & cgh_)
					{
						auto access_a = buff_a.get_access<sycl::access::mode::read>(cgh_);
						auto access_b = buff_b.get_access<sycl::access::mode::read>(cgh_);
						auto access_c = buff_c.get_access<sycl::access::mode::write>(cgh_);

						cgh_.parallel_for<class vector_add>(
							work_items,
							[=, data = vals.data()](sycl::id<1> tid_)
							{
								access_c[tid_] =
									// NOLINTNEXTLINE(*-pro-bounds-pointer-arithmetic)
									access_a[tid_] + access_b[tid_] + data[0] + data[1];
							});
					});
				queue.wait_and_throw();
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {3.F, 7.F, 3.F, 11.F, 3.F};

				CHECK(c == expected);
			}
		}

		WHEN("USM vector is doubled by kernel")
		{
			using Allocator = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
			auto vals = felt2::device::make_unique_sycl<std::vector<float, Allocator>>(
				queue.get_device(), queue.get_context(), Allocator{queue});
			vals->push_back(1);
			vals->push_back(2);
			sycl::range<1> work_items{vals->size()};
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_double>(
						work_items,
						[pvals = vals.get()](sycl::id<1> tid_)
						{
							auto & vals = *pvals;
							auto const val = vals[tid_];
							vals[tid_] = val * static_cast<float>(vals.size());
						});
				});
			queue.wait_and_throw();

			THEN("values have been doubled")
			{
				std::vector<float, Allocator> expected{Allocator{queue}};
				expected.push_back(2);
				expected.push_back(4);

				CHECK(*vals == expected);
			}
		}
		WHEN("CUDA error")
		{
			using Allocator = sycl::usm_allocator<char, sycl::usm::alloc::shared>;
			std::vector<char, Allocator> vals(Allocator{queue});
			static constexpr std::size_t k_buff_len = 20;
			vals.resize(k_buff_len);

			sycl::range<1> work_items{vals.size()};
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_double>(
						work_items,
						[buff = vals.data()](sycl::id<1> tid_) {
							std::format_to_n(
								buff, k_buff_len, "Hello from thread {}", static_cast<int>(tid_));
						});
				});

			THEN("Error is reported")
			{
				try
				{
					queue.wait_and_throw();
					FAIL("Should have thrown");
				}
				catch (const std::exception & exc)
				{
					std::string const error_message{exc.what()};
					CHECK_THAT(
						error_message,
						Catch::Matchers::ContainsSubstring(
							"Unresolved extern function 'memchr' (error code = CU:218)"));
				}
			}
		}
	}
}
#pragma clang diagnostic pop

SCENARIO("Basic oneMKL usage")
{
	GIVEN("Input vectors")
	{
		std::vector<float> a = {1.F, 2.F, 3.F, 4.F, 5.F};
		std::vector<float> b = {-1.F, 2.F, -3.F, 4.F, -5.F};
		assert(a.size() == b.size());

		WHEN("vectors are added using oneMKL")
		{
			{
				sycl::gpu_selector const selector;
				sycl::queue queue{selector, &async_handler};  // NOLINT(misc-const-correctness)
				sycl::buffer<float> buff_a(a.data(), a.size());
				sycl::buffer<float> buff_b(b.data(), b.size());

				// NOTE: if a segfault happens here it's because the ERROR_MSG is nullptr, which
				// means there are no enabled backend libraries.
				oneapi::mkl::blas::column_major::axpy(
					queue, static_cast<std::int64_t>(a.size()), 1.0F, buff_a, 1, buff_b, 1);
			}
			THEN("result is as expected")
			{
				std::vector<float> expected = {0.F, 4.F, 0.F, 8.F, 0.F};

				CHECK(b == expected);
			}
		}
	}
}

template <class String, class... Args>
void append(String & str_, Args... args_)
{
	(
		[&]
		{
			if constexpr (requires { std::string_view{args_}; })
			{
				str_ += args_;
			}
			else
			{
				etl::to_string(args_, str_, true);
			}
		}(),
		...);
}

struct ThreadAppender
{
	sycl::private_ptr<std::size_t const> tid;

	template <class... Args>
	void append(etl::string_ext & str_, Args... args_) const
	{
		etl::to_string(*tid, str_, true);
		str_ += ": ";
		(
			[&]
			{
				if constexpr (requires { std::string_view{args_}; })
				{
					str_ += args_;
				}
				else
				{
					etl::to_string(args_, str_, true);
				}
			}(),
			...);
	}
};

SCENARIO("Assertion and logging in SYCL")
{
	GIVEN("queue and work range")
	{
		sycl::queue queue{sycl::gpu_selector_v, &async_handler};  // NOLINT(misc-const-correctness)

		static constexpr std::size_t k_num_work_items = 5;
		static constexpr std::size_t k_max_msg_size = 30;

		sycl::range<1> work_items{k_num_work_items};

		using Allocator = sycl::usm_allocator<char, sycl::usm::alloc::shared>;
		using Extents = stdex::extents<char, std::dynamic_extent, k_max_msg_size>;
		std::vector<char, Allocator> text_data(
			k_num_work_items * k_max_msg_size, '\0', Allocator{queue});
		stdex::mdspan text_nd{text_data.data(), Extents{k_num_work_items}};

		WHEN("a kernel with logging is executed")
		{
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[text_nd](sycl::id<1> tid_)
						{
							etl::string_ext text{
								&text_nd[static_cast<int>(tid_), 0], k_max_msg_size};
							append(text, "Hello from thread ", static_cast<int>(tid_));
						});
				});
			queue.wait_and_throw();
			THEN("log is output")
			{
				for (int tid = 0; tid < text_nd.extents().extent(0); ++tid)
				{
					std::string_view const msg{&text_nd[tid, 0]};

					CHECK(msg == fmt::format("Hello from thread {}", tid));
				}
			}
		}
		WHEN("a kernel with a logger that relies on private memory is executed")
		{
			auto appender = felt2::device::make_unique_sycl<ThreadAppender>(
				1, queue.get_device(), queue.get_context());

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[text_nd, appender = appender.get()](sycl::id<1> tid_)
						{
							std::size_t const tidsz = tid_;
							appender->tid = &tidsz;

							etl::string_ext text{
								&text_nd[static_cast<int>(tid_), 0], k_max_msg_size};
							appender->append(text, "Hello from thread ", static_cast<int>(tid_));
						});
				});
			queue.wait_and_throw();

			THEN("log is output")
			{
				for (int tid = 0; tid < text_nd.extents().extent(0); ++tid)
				{
					std::string_view const msg{&text_nd[tid, 0]};

					CHECK(msg == fmt::format("{}: Hello from thread {}", tid, tid));
				}
			}
		}
		WHEN("a kernel with a logger is executed")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{
							auto const stream_id = static_cast<std::size_t>(tid_);
							logger.set_stream(&stream_id);
							logger.log("Hello from thread ", stream_id);
						});
				});
			queue.wait_and_throw();
			THEN("log is output")
			{
				for (std::size_t tid = 0; tid < work_items.get(0); ++tid)
				{
					CHECK(logger.text(tid) == fmt::format("Hello from thread {}", tid));
				}
			}
		}
		WHEN("a kernel logs to unexpected stream ids")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{
							auto const stream_id = static_cast<std::size_t>(2 * tid_);
							logger.set_stream(&stream_id);
							logger.log("Hello from thread ", static_cast<int>(tid_), "\n");
						});
				});
			queue.wait_and_throw();

			THEN("log output stream is a circular buffer")
			{
				for (std::size_t tid = 0; tid < work_items.get(0); ++tid)
				{
					CAPTURE(tid);
					CHECK(
						logger.text((2 * tid) % work_items.get(0)) ==
						fmt::format("Hello from thread {}\n", tid));
				}
			}
		}

		WHEN("a kernel logs to thread-local stream and no stream is set")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{ logger.log("Hello from thread ", static_cast<int>(tid_), "\n"); });
				});
			queue.wait_and_throw();

			THEN("all logs go to stream zero")
			{
				CAPTURE(logger.text(0));
				CHECK_THAT(
					std::string{logger.text(0)},
					Catch::Matchers::StartsWith(fmt::format("Hello from thread")));

				for (std::size_t tid = 1; tid < work_items.get(0); ++tid)
					CHECK(logger.text(tid).empty());
			}
		}

		WHEN("a kernel logs to thread-local stream then again with no stream set")
		{
			auto storage = felt2::components::device::Log::make_storage(
				queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
			felt2::components::device::Log logger;
			logger.set_storage(storage);

			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{
							std::size_t const stream_id = tid_.get(0);
							logger.set_stream(&stream_id);

							logger.log("Hello from thread ", static_cast<int>(tid_), "\n");
						});
				});
			queue.wait_and_throw();
			queue.submit(
				[&](sycl::handler & cgh_)
				{
					cgh_.parallel_for<class vector_add>(
						work_items,
						[logger](sycl::id<1> tid_)
						{ logger.log("Hello again from thread ", static_cast<int>(tid_), "\n"); });
				});
			queue.wait_and_throw();

			THEN("second set of logs go to stream zero")
			{
				CAPTURE(logger.text(0));
				CHECK_THAT(
					std::string{logger.text(0)},
					Catch::Matchers::StartsWith(
						fmt::format("Hello from thread 0\nHello again from thread")));

				for (std::size_t tid = 1; tid < work_items.get(0); ++tid)
					CHECK(
						logger.text(tid) ==
						std::format("Hello from thread {}\n", static_cast<int>(tid)));
			}
		}
	}
}

SCENARIO("SyCL with ConvGrid")
{
	GIVEN("Shared grid")
	{
		sycl::context const ctx;
		sycl::device const dev{sycl::gpu_selector_v};
		using ConvGrid = convfelt::ConvGridTD<float, 3, convfelt::GridFlag::is_device_shared>;

		auto pgrid = felt2::device::make_unique_sycl<ConvGrid>(
			dev,
			ctx,
			convfelt::make_device_context(dev, ctx),
			felt2::Vec3i{4, 4, 3},
			felt2::Vec2i{2, 2});

		std::fill(pgrid->storage().begin(), pgrid->storage().end(), 3);
		REQUIRE(!pgrid->children().storage().empty());
		CHECK(pgrid->children().storage().size() == 4);
		CHECK(pgrid->children().storage()[0].storage().size() > 1);
		CHECK(pgrid->children().storage()[0].storage().data() == pgrid->storage().data());

		WHEN("grid data is doubled using sycl")
		{
			sycl::range<1> const work_items{pgrid->children().storage().size()};

			sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

			[[maybe_unused]] auto const log_storage = felt2::components::device::Log::make_storage(
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
			THEN("result is as expected")
			{
				CHECK(!pgrid->context().logger().has_logs());

				for (auto const val : convfelt::iter::val(*pgrid)) CHECK(val == 6);
			}
		}

		WHEN("out of bounds access")
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
				CHECK(pgrid->context().logger().text(0UZ).empty());
				CHECK(pgrid->context().logger().text(1UZ).empty());
				CHECK(pgrid->context().logger().text(2UZ).empty());
				CHECK(
					pgrid->context().logger().text(3UZ) ==
					"AssertionError: get:  assert_pos_idx_bounds(4) i.e. (0, 0, 0) is greater than "
					"extent 4\n");
			}
		}
	}
}

SCENARIO("Applying filter to ConvGrid")
{
	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{k_file_path}}; // NOLINT(misc-const-correctness)
		image.read();
		auto image_grid_spec = image.spec();
		image_grid_spec.width += 2;
		image_grid_spec.height += 2;
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

		// False positive:
		// NOLINTNEXTLINE(misc-const-correctness)
		convfelt::InputGrid image_grid{
			convfelt::make_host_context(),
			{image.spec().height + 2, image.spec().width + 2, image_grid_spec.nchannels},
			{0, 0, 0},
			0};

		// NOLINTNEXTLINE(misc-const-correctness)
		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		AND_GIVEN("image is split into filter regions")
		{
			sycl::context const ctx;
			sycl::device const dev{sycl::gpu_selector_v};
			//			sycl::device dev{sycl::cpu_selector_v};
			using FilterGrid =
				convfelt::ConvGridTD<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>;

			FilterSizeHelper const filter_input_sizer{felt2::Vec3i{4, 4, 3}, felt2::Vec3i{2, 2, 0}};

			auto input_size = filter_input_sizer.input_size_from_source_size(image_grid.size());

			auto filter_input_grid = felt2::device::make_unique_sycl<FilterGrid>(
				dev,
				ctx,
				convfelt::make_device_context(dev, ctx),
				input_size,
				filter_input_sizer.filter_window());

			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid->children()))
			{
				const felt2::Vec3i input_pos_start =
					filter_input_sizer.input_start_pos_from_filter_pos(
						filter_input_grid->children().index(filter_pos_idx));

				for (felt2::PosIdx const local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					const felt2::Vec3i input_pos =
						input_pos_start + felt2::index(local_pos_idx, filter.size());

					filter.set(local_pos_idx, image_grid.get(input_pos));
				}
			}

			WHEN("image is split into filter regions on device")
			{
				auto image_grid_device = felt2::device::make_unique_sycl<
					convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
					dev,
					ctx,
					convfelt::make_device_context(dev, ctx),
					image_grid.size(),
					image_grid.offset(),
					0.0F);

				image_grid_device->storage().assign(
					image_grid.storage().begin(), image_grid.storage().end());

				auto filter_input_grid_device = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convfelt::make_device_context(dev, ctx),
					input_size,
					filter_input_sizer.filter_window());
				sycl::range<1> const work_items{image_grid_device->storage().size()};
				sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

				[[maybe_unused]] auto const log_storage =
					felt2::components::device::Log::make_storage(
						queue.get_device(), queue.get_context(), work_items.get(0), 1024UL);
				image_grid_device->context().logger().set_storage(log_storage);
				filter_input_grid_device->context().logger().set_storage(log_storage);

				queue.submit(
					[&](sycl::handler & cgh_)
					{
						cgh_.parallel_for<class grid_copy>(
							work_items,
							[filter_input_sizer,
							 image_grid_device = image_grid_device.get(),
							 filter_input_grid_device =
								 filter_input_grid_device.get()](sycl::item<1> item_)
							{
								auto & filter_inputs = filter_input_grid_device->children();

								felt2::PosIdx const input_pos_idx = item_.get_linear_id();
								felt2::Vec3i const source_pos =
									image_grid_device->index(input_pos_idx);

								felt2::Scalar const source_value =
									image_grid_device->get(input_pos_idx);

								filter_input_sizer.input_pos_from_source_pos(
									image_grid_device->size(),
									source_pos,
									[&](felt2::Vec3i const & filter_pos_,
										felt2::Vec3i const & global_pos_) {
										filter_inputs.get(filter_pos_)
											.set(global_pos_, source_value);
									});
							});
					});
				queue.wait_and_throw();
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

					for (felt2::PosIdx const child_pos_idx :
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

			AND_GIVEN("an output image and weight matrix")
			{
				FilterSizeHelper const filter_output_sizer{felt2::Vec3i{2, 2, 4}};

				auto filter_output_grid = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convfelt::make_device_context(dev, ctx),
					filter_output_sizer.output_size_from_num_filter_regions(
						filter_input_grid->children().size()),
					filter_output_sizer.filter_window());

				REQUIRE(
					filter_output_grid->children().size() == filter_input_grid->children().size());

				convfelt::USMMatrix const usm_weights{
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
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					queue.submit([pgrid = filter_input_grid.get()](sycl::handler & cgh_)
								 { cgh_.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					queue.submit([pgrid = filter_output_grid.get()](sycl::handler & cgh_)
								 { cgh_.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					queue.submit(
						[&usm_weights](sycl::handler & cgh_)
						{ cgh_.prefetch(usm_weights.bytes().data(), usm_weights.bytes().size()); });

					sycl::nd_range<1> const work_range{
						filter_output_grid->storage().size(),
						static_cast<size_t>(filter_output_grid->child_size().prod())};

					static constexpr auto k_max_log_len = 1024UZ;

					auto log_storage = felt2::components::device::Log::make_storage(
						queue.get_device(),
						queue.get_context(),
						work_range.get_local().get(0),
						k_max_log_len);
					filter_input_grid->context().logger().set_storage(log_storage);
					filter_output_grid->context().logger().set_storage(log_storage);

					queue.submit(
						[&](sycl::handler & cgh_)
						{
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
								sycl::access::target::local> const input_child_data{
								static_cast<std::size_t>(usm_weights.matrix().cols()), cgh_};

							cgh_.parallel_for<class grid_mult>(
								work_range,
								[filter_input_grid = filter_input_grid.get(),
								 filter_output_grid = filter_output_grid.get(),
								 weights = usm_weights.matrix(),
								 input_child_data](sycl::nd_item<1> item_)
								{
									std::size_t const group_id = item_.get_group_linear_id();
									std::size_t const local_id = item_.get_local_linear_id();

									auto & input_child =
										filter_input_grid->children().get(group_id);
									auto & output_child =
										filter_output_grid->children().get(group_id);

									if (input_child.storage().size() != input_child_data.size())
									{
										filter_input_grid->context().logger().log(
											local_id,
											"input_child.storage().size() != "
											"input_child_data.size() i.e. ",
											input_child.storage().size(),
											" != ",
											input_child_data.size(),
											"\n");
										return;
									}
									if (local_id >= output_child.storage().size())
									{
										filter_output_grid->context().logger().log(
											local_id,
											"local_id >= output_child.storage().size() i.e. ",
											local_id,
											" >= ",
											output_child.storage().size(),
											"\n");
										return;
									}

									// TODO(DF): Do we need to construct filter input regions in
									//  global memory only to copy out again into local memory?
									//  I.e. will we need the input grid further down the line?
									//  Otherwise could construct input region here - see next
									//  test.
									item_
										.async_work_group_copy(
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
									ColVectorMap output_vec{// NOLINT(misc-const-correctness)
															output_child.storage().data(),
															weights.rows()};

									auto const row_idx = static_cast<Eigen::Index>(local_id);
									output_vec(row_idx) = weights.row(row_idx).dot(input_vec);
								});
						});
					queue.wait_and_throw();

					THEN("values are as expected")
					{
						CHECK(!filter_input_grid->context().logger().has_logs());
						CHECK(!filter_output_grid->context().logger().has_logs());
						assert_expected_values();
					}
				}  // WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")

				WHEN("filter is applied in kernel that computes filter input on the fly")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					auto filter_input_template = felt2::device::make_unique_sycl<
						convfelt::TemplateParentGridTD<felt2::Scalar, 3, 1U>>(
						queue.get_device(),
						queue.get_context(),
						convfelt::make_device_context(queue),
						input_size,
						filter_input_sizer.filter_window());

					auto image_grid_device = felt2::device::make_unique_sycl<
						convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
						queue.get_device(),
						queue.get_context(),
						convfelt::make_device_context(queue),
						image_grid.size(),
						image_grid.offset(),
						0.0F);

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

					constexpr auto k_scalar = [](auto val_)
					{ return static_cast<felt2::Scalar>(val_); };

					felt2::Scalar const ideal_work_groups_per_image =
						k_scalar(elems_per_image) / k_scalar(ideal_elems_per_work_group);

					felt2::Scalar const ideal_filters_per_work_group =
						k_scalar(filters_per_image) / ideal_work_groups_per_image;

					auto const filters_per_work_group =
						static_cast<std::size_t>(std::ceil(ideal_filters_per_work_group));

					felt2::Scalar const work_groups_per_image =
						k_scalar(filters_per_image) / k_scalar(filters_per_work_group);

					felt2::Scalar const elems_per_work_group =
						k_scalar(elems_per_image) / work_groups_per_image;

					auto const num_work_groups =
						static_cast<std::size_t>(std::ceil(work_groups_per_image));
					auto const work_group_size = std::min(
						static_cast<std::size_t>(std::ceil(elems_per_work_group)), elems_per_image);

					CAPTURE(
						dev.get_info<sycl::info::device::local_mem_size>(),
						dev.get_info<sycl::info::device::sub_group_sizes>());
					CHECK(
						dev.get_info<sycl::info::device::local_mem_size>() > sizeof(felt2::Scalar) *
							static_cast<std::size_t>(filter_input_sizer.num_filter_elems()));
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

					sycl::nd_range<1> const work_range{
						num_work_groups * work_group_size, work_group_size};

					[[maybe_unused]] auto const log_storage =
						felt2::components::device::Log::make_storage(
							queue.get_device(),
							queue.get_context(),
							work_range.get_local().get(0),
							1024);

					filter_input_template->context().logger().set_storage(log_storage);
					filter_input_grid->context().logger().set_storage(log_storage);
					filter_output_grid->context().logger().set_storage(log_storage);
					image_grid_device->context().logger().set_storage(log_storage);

					queue.submit(
						[&](sycl::handler & cgh_)
						{
							sycl::local_accessor<felt2::Scalar, 2> const input_child_data{
								sycl::range(
									filters_per_work_group,
									static_cast<std::size_t>(usm_weights.matrix().cols())),
								cgh_};

							cgh_.parallel_for<class dynamic_input>(
								work_range,
								[image_grid = image_grid_device.get(),
								 filter_input_template = filter_input_template.get(),
								 filter_output_grid = filter_output_grid.get(),
								 filters_per_work_group,
								 weights = usm_weights.matrix(),
								 input_child_data,
								 filter_input_sizer,
								 input_size = filter_input_sizer.input_size_from_source_size(
									 image_grid.size())](sycl::nd_item<1> item_)
								{
									std::size_t const group_id = item_.get_group_linear_id();
									std::size_t const local_id = item_.get_local_linear_id();

									felt2::PosIdx const elems_per_work_group =
										item_.get_local_range(0);
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
									// TODO(DF): For large filters (e.g. fully connected layers)
									//  should fall back to global memory?  Or refactor kernel
									//  somehow?  For fully connected layer, input_child is the same
									//  image_grid, so no need to sample into local memory. So
									//  perhaps safe to assume either that fully connected case or
									//  small-enough filters for local memory?
									input_child.storage() = std::span(
										&input_child_data[filter_idx_offset_for_item][0], num_cols);

									auto & output_child =
										filter_output_grid->children().get(filter_idx_for_item);

									// Assert invariants.
									if (num_cols != input_child.storage().size())
									{
										filter_input_template->context().logger().log(
											local_id,
											"num_cols != input_child.storage().size() i.e. ",
											num_cols,
											" != ",
											input_child.storage().size(),
											"\n");
										return;
									}
									if (static_cast<felt2::PosIdx>(weights.rows()) >
										elems_per_filter)
									{
										filter_output_grid->context().logger().log(
											local_id,
											"weights.rows()	> elems_per_filter i.e. ",
											weights.rows(),
											" > ",
											elems_per_filter,
											"\n");
										return;
									}

									// Seriously?
									// NOLINTNEXTLINE(misc-const-correctness)
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

									item_.barrier(sycl::access::fence_space::local_space);

									auto const row_idx =
										static_cast<Eigen::Index>(elem_idx_for_item);

									output_child.matrix()(row_idx) =
										weights.row(row_idx).dot(input_child.matrix());
								});
						});
					queue.wait_and_throw();

					THEN("values are as expected")
					{
						CHECK(!filter_input_grid->context().logger().has_logs());
						CHECK(!filter_output_grid->context().logger().has_logs());
						CHECK(!image_grid_device->context().logger().has_logs());
						CHECK(!filter_input_template->context().logger().has_logs());
						assert_expected_values();
					}
				}  // WHEN("filter is applied in kernel that computes filter input on the fly")

				WHEN("filter is applied to grid using oneMKL")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					oneapi::mkl::blas::column_major::gemm(
						queue,
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

					queue.wait_and_throw();
					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}
			}  // AND_GIVEN("an output image and weight matrix")

			AND_GIVEN("an output vector and weight matrix")
			{
				FilterSizeHelper const filter_output_sizer{felt2::Vec3i(1, 1, 5)};

				auto filter_output_grid = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convfelt::make_device_context(dev, ctx),
					filter_output_sizer.output_size_from_num_filter_regions(felt2::Vec3i::Ones()),
					filter_output_sizer.filter_window());

				REQUIRE(filter_output_grid->children().size() == felt2::Vec3i(1, 1, 1));

				convfelt::USMMatrix const usm_weights{
					filter_output_sizer.num_filter_elems(), image_grid.size().prod(), dev, ctx};

				usm_weights.matrix().setRandom();

				auto const assert_expected_values = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid->children()))
					{
						const felt2::Vec3i filter_pos =
							filter_input_grid->children().index(filter_pos_idx);
						CAPTURE(filter_pos);

						auto const input_matrix = image_grid.matrix();
						auto const output_matrix = filter_output_grid->matrix();
						CAPTURE(usm_weights.matrix() * input_matrix, output_matrix);
						auto const diff = usm_weights.matrix() * input_matrix - output_matrix;
						felt2::Scalar const diff_sq = diff.dot(diff);
						CHECK(diff_sq == Catch::Approx(0).margin(1e-6));
					}
				};

				WHEN("filter is applied to grid using oneMKL")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					auto image_grid_device = felt2::device::make_unique_sycl<
						convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
						dev,
						ctx,
						convfelt::make_device_context(dev, ctx),
						image_grid.size(),
						image_grid.offset(),
						0.0F);

					image_grid_device->storage().assign(
						image_grid.storage().begin(), image_grid.storage().end());

					oneapi::mkl::blas::column_major::gemm(
						queue,
						oneapi::mkl::transpose::nontrans,
						oneapi::mkl::transpose::nontrans,
						// m: Rows of output / rows of weights
						filter_output_grid->matrix().rows(),
						// n: Columns of output / columns of input
						filter_output_grid->matrix().cols(),
						// k: Rows of input / columns of weights
						image_grid_device->matrix().rows(),
						// alpha
						1,
						// a: weights
						usm_weights.matrix().data(),
						usm_weights.matrix().rows(),
						// b: input
						image_grid_device->matrix().data(),
						image_grid_device->matrix().rows(),
						// beta
						0,
						// c: output
						filter_output_grid->matrix().data(),
						filter_output_grid->matrix().rows());

					queue.wait_and_throw();
					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}
			}
		}  // AND_GIVEN("image is split into filter regions")
	}
}
// NOLINTEND(readability-identifier-length)
// NOLINTEND(*-magic-numbers)
// NOLINTEND(readability-function-cognitive-complexity)
