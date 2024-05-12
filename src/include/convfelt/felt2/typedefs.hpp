// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

/// Format Eigen vectors as row vectors, i.e. "(1,3,2)".
// #define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, DontAlignCols, " ", ",", "", "", "(", ")")

#include <bit>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef FELT2_DEBUG_ENABLED
#ifndef FELT2_DEBUG_DISABLED
#define FELT2_DEBUG_ENABLED NDEBUG
#endif
#endif

#ifdef FELT2_DEBUG_ENABLED
#define FELT2_DEBUG(...) __VA_ARGS__;
#define FELT2_DEBUG_CALL(obj) obj
#else
#define FELT2_DEBUG(...)
#define FELT2_DEBUG_CALL(obj) \
	if (true)                 \
	{                         \
	}                         \
	else                      \
		obj
#endif

namespace felt2
{
using Signed = int32_t;
using Unsigned = std::size_t;

/**
 * Scalar.
 */
using Scalar = float;
/**
 * Position along an axis
 */
using AxisPos = Signed;
/**
 * Grid dimension type.
 */
using Dim = Signed;
/**
 * Index of a position in a grid's data array.
 */
using PosIdx = Unsigned;
/**
 * Shorthand for D-dimensional vector with elements of T type.
 */
template <typename T, Dim D>
using VecDT = Eigen::Matrix<T, D, 1>;
/**
 * Shorthand for D-dimensional Distance vector.
 */
template <Dim D>
using VecDf = VecDT<Scalar, D>;
/**
 * Shorthand for D-dimensional NodeIdx vector.
 */
template <Dim D>
using VecDi = VecDT<Signed, D>;
/**
 * Shorthand for D-dimensional unsigned integer vector.
 */
template <Dim D>
using VecDu = VecDT<Unsigned, D>;
/**
 * Shorthand for 2D float vector.
 */
using Vec2f = VecDf<2>;
/**
 * Shorthand for 2D unsigned integer vector.
 */
using Vec2u = VecDu<2>;
/**
 * Shorthand for 2D integer vector.
 */
using Vec2i = VecDi<2>;
/**
 * Shorthand for 3D float vector.
 */
using Vec3f = VecDf<3>;
/**
 * Shorthand for 3D unsigned integer vector.
 */
using Vec3u = VecDu<3>;
/**
 * Shorthand for 3D integer vector.
 */
using Vec3i = VecDi<3>;

/**
 * Strong type for power-of-two vector.
 *
 * @tparam T Element type.
 * @tparam D Dimensions.
 */
template <Dim D>
class PowTwoDu
{
public:
	using Vec = VecDu<D>;

	constexpr static PowTwoDu<D> from_minimum_size(Vec const & desired_size_)
	{
		Vec const exponents = desired_size_.unaryExpr(
			[](auto const value_)
			{
				return static_cast<typename Vec::value_type>(std::bit_width(value_)) -
					std::has_single_bit(value_);
			});
		return PowTwoDu<D>{exponents};
	}

	constexpr static PowTwoDu<D> from_exponents(Vec const & exponents_)
	{
		return PowTwoDu<D>{exponents_};
	}

	friend bool operator==(PowTwoDu<D> const & lhs_, PowTwoDu<D> const & rhs_)
	{
		return lhs_.m_exponents == rhs_.m_exponents;
	}

	constexpr auto vec() const
	{
		return m_exponents.unaryExpr([](auto const exponent_) { return 1UL << exponent_; });
	}

	constexpr Vec const & exps() const
	{
		return m_exponents;
	}

	constexpr auto mask() const
	{
		return m_exponents.unaryExpr([](auto const exponent_) { return (1UL << exponent_) - 1; });
	}

private:
	explicit PowTwoDu(Vec const & exponents_) : m_exponents{exponents_} {}
	Vec m_exponents;
};

}  // namespace felt2
