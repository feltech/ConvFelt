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
	using ColArrayMap = Eigen::Map<Eigen::Array<Leaf, 1, Eigen::Dynamic>, Eigen::ColMajor>;
	using ColVectorMap = Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, 1>, Eigen::ColMajor>;

	std::reference_wrapper<Storage> m_storage_impl;

	/**
	 * Map the raw data to a (column-major) Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	constexpr ColArrayMap array() noexcept
	{
		return ColArrayMap(
			m_storage_impl.get().storage().data(),
			Eigen::Index(m_storage_impl.get().storage().size()));
	}
	/**
	 * Map the raw data to a (row-major) Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	constexpr ColVectorMap matrix() noexcept
	{
		return ColVectorMap(
			m_storage_impl.get().storage().data(),
			Eigen::Index(m_storage_impl.get().storage().size()));
	}

	[[nodiscard]] constexpr ColVectorMap matrix() const noexcept
	{
		return ColVectorMap(
			m_storage_impl.get().storage().data(),
			Eigen::Index(m_storage_impl.get().storage().size()));
	}
};

template <HasLeafType Traits, HasStorage Data, HasChildrenSize ChildrenSize>
struct MatrixColPerChild
{
	/// Type of data to store in grid nodes.
	using Leaf = typename Traits::Leaf;
	using MatrixMap =
		Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

	std::reference_wrapper<Data> m_data_impl;
	std::reference_wrapper<ChildrenSize> m_children_size_impl;

	/**
	 * Map the raw data to a  Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	MatrixMap matrix() noexcept
	{
		return {
			m_data_impl.get().storage().data(),
			Eigen::Index(m_children_size_impl.get().num_elems_per_child()),
			Eigen::Index(m_children_size_impl.get().num_children())};
	}

	[[nodiscard]] MatrixMap matrix() const noexcept
	{
		return {
			m_data_impl.get().storage().data(),
			Eigen::Index(m_children_size_impl.get().num_elems_per_child()),
			Eigen::Index(m_children_size_impl.get().num_children())};
	}
};

template <HasStorage Storage>
struct MatrixMap
{
	/// Type of data to store in grid nodes.
	using Leaf = StorageLeafT<Storage>;
	using Matrix = Eigen::Map<Eigen::Matrix<Leaf, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

	felt2::Dim m_rows;
	felt2::Dim m_cols;
	std::reference_wrapper<Storage> m_storage_impl;

	/**
	 * Map the raw data to Eigen::Map, which can be used for BLAS arithmetic.
	 *
	 * @return Eigen compatible vector of data array.
	 */
	[[nodiscard]] constexpr Matrix matrix() noexcept
	{
		return {m_storage_impl.get().storage().data(), m_rows, m_cols};
	}

	[[nodiscard]] constexpr Matrix matrix() const noexcept
	{
		return {m_storage_impl.get().storage().data(), m_rows, m_cols};
	}
};

}  // namespace felt2::components