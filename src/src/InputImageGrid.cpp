// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <convfelt/InputImageGrid.hpp>
#include <utility>

convfelt::InputImageGrid::HostGrid convfelt::InputImageGrid::make_from_file(
	std::filesystem::path path)
{
	return convfelt::InputImageGrid::HostGrid{
		make_host_context(), felt2::Vec3i(0, 0, 0), felt2::Vec3i(0, 0, 0), 0};
}

convfelt::InputImageGrid::HostDeviceGrid convfelt::InputImageGrid::make_from_file(
	std::filesystem::path path, sycl::context context, sycl::device device)
{
	return convfelt::InputImageGrid::HostDeviceGrid{
		make_device_context(device, context), felt2::Vec3i(0, 0, 0), felt2::Vec3i(0, 0, 0), 0};
}
