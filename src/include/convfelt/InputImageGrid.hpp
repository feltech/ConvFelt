// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>

#include <sycl/sycl.hpp>

#include "ConvGrid.hpp"
#include "felt2/typedefs.hpp"

namespace convfelt
{
struct InputImageGrid
{
	using HostGrid = ByValue<felt2::Scalar, 3>;
	using HostDeviceGrid = ByValue<felt2::Scalar, 3, GridFlag::is_device_shared>;

	[[nodiscard]] static HostGrid make_from_file(std::filesystem::path const & path_);
	[[nodiscard]] static HostDeviceGrid make_from_file(
		std::filesystem::path const & path_,
		sycl::context const & context_,
		sycl::device const & device_);
};
}  // namespace convfelt