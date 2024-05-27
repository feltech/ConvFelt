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
 * Strong type for a power-of-two vector.
 *
 * This class represents vectors with power-of-two dimension values.
 *
 * @tparam D Number of dimensions of the vector.
 */
template <Dim D>
class PowTwoDu
{
public:
	using Vec = VecDu<D>;

	/**
	 * Creates a PowTwoDu from a minimum desired size.
	 *
	 * The returned object represents the minimum power-of-two vector that is required to achieve
	 * the desired size.
	 *
	 * @tparam Derived The derived type of the Eigen matrix.
	 * @param desired_size_ The minimum desired size.
	 * @return Power-of-two vector.
	 */
	template <typename Derived>
	constexpr static PowTwoDu<D> from_minimum_size(Eigen::MatrixBase<Derived> const & desired_size_)
	{
		Vec const exponents = desired_size_.unaryExpr(
			[](auto const value_)
			{
				auto const uvalue = static_cast<Unsigned>(value_);
				return static_cast<typename Vec::value_type>(std::bit_width(uvalue)) -
					std::has_single_bit(uvalue);
			});
		return PowTwoDu<D>{exponents};
	}


	/**
	 * Creates a PowTwoDu object from a minimum desired size.
	 *
	 * The PowTwoDu object represents the power-of-two vector that is required to achieve the
	 * desired size.
	 *
	 * Shorthand signature such that from_minimum_size({x, y, z}) can be used, i.e. without needing
	 * to specify type of input vector so that the Derived template paramter can be deduced.
	 *
	 * @param desired_size_ The minimum desired size.
	 * @return Power-of-two vector.
	 */
	constexpr static PowTwoDu<D> from_minimum_size(VecDi<D> const & desired_size_)
	{
		return from_minimum_size<VecDi<D>>(desired_size_);
	}

	/**
	 * Creates a PowTwoDu object from a vector of exponents.
	 *
	 * @param exponents_ Vector of exponents representing the power-of-two values.
	 * @return Power-of-two vector.
	 */
	constexpr static PowTwoDu<D> from_exponents(Vec const & exponents_)
	{
		return PowTwoDu<D>{exponents_};
	}

	/**
	 * Equality operator.
	 *
	 * Two PowTwoDu object compare equal if they have the same exponents.
	 */
	friend bool operator==(PowTwoDu<D> const & lhs_, PowTwoDu<D> const & rhs_)
	{
		return lhs_.m_exponents == rhs_.m_exponents;
	}

	/**
	 * Get as a linear unsigned vector.
	 *
	 * @return An Eigen expression that evaluates to a linear unsigned vector.
	 */
	constexpr auto as_size() const
	{
		return m_exponents.unaryExpr([](auto const exponent_) { return 1UL << exponent_; });
	}

	/**
	 * Get as a linear signed vector.
	 *
	 * @return An Eigen expression that evaluates to a linear signed vector.
	 */
	constexpr auto as_pos() const
	{
		return m_exponents.unaryExpr(
			[](auto const exponent_)
			{ return static_cast<VecDi<D>::value_type>(1UL << exponent_); });
	}

	/**
	 * Get the vector of exponents representing the power-of-two values.
	 *
	 * @return The vector of exponents.
	 */
	constexpr Vec const & exps() const
	{
		return m_exponents;
	}

	/**
	 * Returns the modulo mask associated with the vector.
	 *
	 * The mask can be used to perform a modulo calculation by applying a bitwise AND operator
	 * between the mask and the target value.
	 *
	 * This modulo mask is useful in optimizing modulo operations, as bitwise AND is faster.
	 *
	 * @return An Eigen expression that computes a bitwise modulo mask when evaluated.
	 */
	constexpr auto mask() const
	{
		return m_exponents.unaryExpr([](auto const exponent_) { return (1UL << exponent_) - 1; });
	}

private:
	explicit PowTwoDu(Vec const & exponents_) : m_exponents{exponents_} {}
	Vec m_exponents;
};
/**
 * Shorthand for 3D power-of-two vector.
 */
using PowTwo3u = PowTwoDu<3>;
}  // namespace felt2
