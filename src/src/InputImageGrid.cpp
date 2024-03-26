// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <filesystem>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/InputImageGrid.hpp>

convfelt::InputImageGrid::HostGrid convfelt::InputImageGrid::make_from_file(
	[[maybe_unused]] std::filesystem::path const & path_)
{
	return convfelt::InputImageGrid::HostGrid{
		make_host_context(), felt2::Vec3i(0, 0, 0), felt2::Vec3i(0, 0, 0), 0};
}

convfelt::InputImageGrid::HostDeviceGrid convfelt::InputImageGrid::make_from_file(
	[[maybe_unused]] std::filesystem::path const & path_,
	sycl::context const & context_,
	sycl::device const & device_)
{
	return convfelt::InputImageGrid::HostDeviceGrid{
		make_device_context(device_, context_), felt2::Vec3i(0, 0, 0), felt2::Vec3i(0, 0, 0), 0};
}
