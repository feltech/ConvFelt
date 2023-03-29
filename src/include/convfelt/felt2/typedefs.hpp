// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

/// Format Eigen vectors as row vectors, i.e. "(1,3,2)".
//#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, DontAlignCols, " ", ",", "", "", "(", ")")

#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef FELT2_DEBUG_DISABLED
#define FELT2_DEBUG_ENABLED NDEBUG
#endif

#ifdef FELT2_DEBUG_ENABLED
#define FELT2_DEBUG(...) __VA_ARGS__;
#else
#define FELT2_DEBUG(...)
#endif

#define FELT2_FWD_COMPONENT_FN(comp, fn)\
decltype(auto) fn(auto &&... args)\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}\
decltype(auto) fn(auto &&... args) const\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}

#define FELT2_FWD_COMPONENT_FN_NOEXCEPT(comp, fn)\
decltype(auto) fn(auto &&... args) noexcept\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}\
decltype(auto) fn(auto &&... args) const noexcept\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}

#define FELT2_FWD_COMPONENT_FN_CONST(comp, fn)\
decltype(auto) fn(auto &&... args) const\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}

#define FELT2_FWD_COMPONENT_FN_CONST_NOEXCEPT(comp, fn)\
decltype(auto) fn(auto &&... args) const noexcept\
{\
	return comp.fn(std::forward<decltype(args)>(args)...);\
}


namespace felt2
{
/**
 * Scalar.
 */
using Scalar = float;
/**
 * Grid dimension type.
 */
using Dim = int;
/**
 * Index of a position in a grid's data array.
 */
using PosIdx = std::size_t;
/**
 * Grid node position vector element.
 */
using NodeIdx = int;
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
using VecDi = VecDT<NodeIdx, D>;
/**
 * Shorthand for D-dimensional unsigned integer vector.
 */
template <Dim D>
using VecDu = VecDT<unsigned, D>;

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
}
