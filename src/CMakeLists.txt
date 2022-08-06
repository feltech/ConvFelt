# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT
add_library(convfelt INTERFACE)

target_compile_features(
	convfelt
	INTERFACE
	cxx_std_20
)


target_include_directories(
	convfelt
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_system_libraries(convfelt INTERFACE
	range-v3::range-v3
	Eigen3::Eigen
	ViennaCL::viennacl
	cppcoro::cppcoro
	hipSYCL::hipSYCL-rt
	felt)

target_compile_definitions(
	convfelt
	INTERFACE
	VIENNACL_WITH_EIGEN
	VIENNACL_WITH_OPENCL
	CL_TARGET_OPENCL_VERSION=220
)

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