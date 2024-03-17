// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>

#include <sycl/sycl.hpp>

#include <convfelt/ConvGrid.hpp>

namespace convfelt
{
struct InputImageGrid
{
	using HostGrid = ByValue<felt2::Scalar, 3, false>;
	using HostDeviceGrid = ByValue<felt2::Scalar, 3, true>;

	[[nodiscard]] static HostGrid make_from_file(std::filesystem::path path);
	[[nodiscard]] static HostDeviceGrid make_from_file(
		std::filesystem::path path, sycl::context context, sycl::device device);
};
}  // namespace convfelt