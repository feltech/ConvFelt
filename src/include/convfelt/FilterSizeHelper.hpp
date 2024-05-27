// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>
#include <utility>

#include "felt2/typedefs.hpp"

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
	using PowTwoDu = felt2::PowTwoDu<D>;

	PowTwoDu filter_size;
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
		return filter_size.as_pos().prod();
	}

	[[nodiscard]] felt2::PowTwoDu<D - 1> filter_window() const
	{
		return felt2::PowTwoDu<D - 1>::from_exponents(filter_size.exps().template head<D - 1>());
	}

	[[nodiscard]] PowTwoDu input_size_from_source_size(PowTwoDu const & source_size_) const
	{
		return PowTwoDu::from_minimum_size(input_size_from_source_and_filter_size(
			filter_size.as_pos(), filter_stride, source_size_.as_pos()));
	}

	[[nodiscard]] VecDi source_start_pos_from_filter_pos(VecDi const & filter_pos_) const
	{
		return (filter_pos_.array() * filter_stride.array()).matrix();
	}

	[[nodiscard]] VecDi source_pos_from_input_pos(VecDi const & input_pos_) const
	{
		return source_pos_from_input_pos_and_filter_size(
			filter_size.as_pos(), filter_stride, input_pos_);
	}

	[[nodiscard]] PowTwoDu output_size_from_num_filter_regions(
		VecDi const & num_filter_regions_) const
	{
		return PowTwoDu::from_minimum_size(
			(num_filter_regions_.array() * filter_size.as_pos().array()).matrix());
	}

	void input_pos_from_source_pos(
		VecDi const & source_size_,
		VecDi const & source_pos_,
		IsCallableWithPos auto && callback_) const
	{
		input_pos_from_source_pos_and_filter_size(
			filter_size.as_pos(),
			filter_stride,
			source_size_,
			source_pos_,
			std::forward<decltype(callback_)>(callback_));
	}

	/**
	 * Assuming we wish to construct a grid storing all filter inputs side-by-side, given a source
	 * image size calculate how many distinct regions will need to be stacked side-by-side.
	 *
	 * @param filter_size_ Size of filter to walk across source image.
	 * @param filter_stride_ Size of each step as the filter walks across the source image.
	 * @param source_size_ Size of source image.
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
	 * @param filter_size_ Size of filter to walk across source image.
	 * @param filter_stride_ Size of each step as the filter walks across the source image.
	 * @param source_size_ Size of source image.
	 * @return Required size to store all filter inputs side-by-side.
	 */
	static VecDi input_size_from_source_and_filter_size(
		VecDi const & filter_size_, VecDi const & filter_stride_, VecDi const & source_size_)
	{
		VecDi const num_filter_regions = num_filter_regions_from_source_and_filter_size(
			filter_size_, filter_stride_, source_size_);

		return (num_filter_regions.array() * filter_size_.array()).matrix();
	}

	/**
	 * Assuming a grid of all filter inputs side-by-side, given a position in this grid, get the
	 * corresponding position in the source image that the filter input element was copied from.
	 *
	 * @param filter_size_ Size of filter to walk across source image.
	 * @param filter_stride_ Size of each step as the filter walks across the source image.
	 * @param input_pos_ Position in grid of all filter inputs side-by-side.
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
	 * @param filter_size_ Size of filter to walk across source image.
	 * @param filter_stride_ Size of each step as the filter walks across the source image.
	 * @param source_size_ Size of source image.
	 * @param source_pos_ Position within source image.
	 * @param callback_ Callback to call, passing the positions in the filter input image that
	 * correspond to the @p source_pos_ position in the source image.
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
FilterSizeHelper(felt2::PowTwoDu<D>, Others...) -> FilterSizeHelper<D>;
