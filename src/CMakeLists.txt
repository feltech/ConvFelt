# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT
add_library(convfelt SHARED include/convfelt/ConvGrid.cpp)

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
target_link_system_libraries(convfelt PUBLIC Eigen3::Eigen viennacl)