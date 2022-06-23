# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT
add_library(convfelt INTERFACE)
target_include_directories(
	convfelt
	INTERFACE
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