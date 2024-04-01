# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT
add_library(
	convfelt
	src/InputImageGrid.cpp
	src/network/Network.cpp
)

target_compile_features(
	convfelt
	PUBLIC
	cxx_std_23
)

file(GLOB_RECURSE _headers CONFIGURE_DEPENDS include/*.h include/*.hpp)

target_sources(
	convfelt
	PUBLIC FILE_SET headers
	TYPE HEADERS
	BASE_DIRS include
	FILES ${_headers}
)

target_link_libraries(
	convfelt
	PRIVATE
	project_options
	project_warnings
	PUBLIC
	range-v3::range-v3
	cppcoro::cppcoro
	Eigen3::Eigen
	hipSYCL::acpp-rt
	yaml-cpp::yaml-cpp
	etl::etl
	std::mdspan
	fmt::fmt
)

target_compile_definitions(
	convfelt
	PRIVATE
	CL_TARGET_OPENCL_VERSION=220
	PUBLIC
	#	EIGEN_USE_SYCL=1
	#	EIGEN_UNALIGNED_VECTORIZE=0
	#	EIGEN_FAST_MATH=0
	#	EIGEN_DONT_VECTORIZE=1
	#	EIGEN_MAX_ALIGN_BYTES=0
	EIGEN_NO_DEBUG=1
)

install(
	TARGETS convfelt
	EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
	FILE_SET headers
)