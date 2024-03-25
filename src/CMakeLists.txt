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
	cxx_std_20
)

target_include_directories(
	convfelt
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
	convfelt
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

# TODO(DF): Use FILE_SET
install(
	DIRECTORY
	${CMAKE_CURRENT_LIST_DIR}/include/convfelt
	DESTINATION
	${CMAKE_INSTALL_INCLUDEDIR}
)
install(
	TARGETS convfelt
	EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
)