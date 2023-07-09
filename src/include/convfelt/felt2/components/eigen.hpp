// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Dense>

#include "core.hpp"

namespace felt2::components
{
template <HasLeafType Traits, HasStorage Storage>
struct EigenMap
{
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	/// Map of of POD to Eigen::Array for manipulation using Eigen BLAS methods.
	using ArrayColMap = Eigen::Map<Eigen::Array<Leaf, 1, Eigen::Dynamic>>;
	using MatrixRowMap = Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, 1>>;

	Storage const & m_storage_impl;

	/**
	 * Map the raw data to a (column-major) Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	ArrayColMap array() noexcept
	{
		return ArrayColMap(
			m_storage_impl.storage().data(), Eigen::Index(m_storage_impl.storage().size()));
	}
	/**
	 * Map the raw data to a (row-major) Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	MatrixRowMap matrix() noexcept
	{
		return MatrixRowMap(
			m_storage_impl.storage().data(), Eigen::Index(m_storage_impl.storage().size()));
	}

	MatrixRowMap matrix() const noexcept
	{
		return MatrixRowMap(
			m_storage_impl.storage().data(), Eigen::Index(m_storage_impl.storage().size()));
	}
};

template <HasLeafType Traits, HasStorage Data, HasChildrenSize ChildrenSize>
struct MatrixColPerChild
{
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	using MatrixMap =
		Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

	Data & m_data_impl;
	ChildrenSize const & m_children_size_impl;

	/**
	 * Map the raw data to a  Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	MatrixMap matrix() noexcept
	{
		return {
			m_data_impl.storage().data(),
			Eigen::Index(m_children_size_impl.num_elems_per_child()),
			Eigen::Index(m_children_size_impl.num_children())};
	}

	MatrixMap matrix() const noexcept
	{
		return {
			m_data_impl.storage().data(),
			Eigen::Index(m_children_size_impl.num_elems_per_child()),
			Eigen::Index(m_children_size_impl.num_children())};
	}
};

template <HasStorage Storage>
struct MatrixMap
{
	/// Type of data to store in grid nodes.
	using Leaf = storage_leaf_t<Storage>;
	using Matrix = Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

	felt2::Dim m_rows;
	felt2::Dim m_cols;
	Storage & m_storage_impl;

	/**
	 * Map the raw data to Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	[[nodiscard]] constexpr Matrix matrix() noexcept
	{
		return {m_storage_impl.storage().data(), m_rows, m_cols};
	}

	[[nodiscard]] constexpr Matrix matrix() const noexcept
	{
		return {m_storage_impl.storage().data(), m_rows, m_cols};
	}
};

}  // namespace felt2::components