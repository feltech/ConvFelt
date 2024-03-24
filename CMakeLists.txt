# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.26)

# Helper CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#------------------------------------------------------------
# Begin project definition.

project(ConvFelt VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)

if (CONVFELT_ENABLE_TESTS)
	# Tests install the project first, so default the install directory to be relative to the build
	# root. This must come before `dynamic_project_options()`
	if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND PROJECT_IS_TOP_LEVEL)
		set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dist"
			CACHE PATH "Installation location" FORCE)
	endif ()
endif ()

# For `generate_export_header`.
include(GenerateExportHeader)
# For `write_basic_package_version_file` et al.
include(CMakePackageConfigHelpers)
# For default install directories, e.g. CMAKE_INSTALL_LIBDIR.
include(GNUInstallDirs)
# Default to less spammy install status messages.
if (NOT DEFINED CMAKE_INSTALL_MESSAGE)
	set(CMAKE_INSTALL_MESSAGE LAZY)
endif ()

#------------------------------------------------------------
# Third-party settings/packages

include(Thirdparty)

#------------------------------------------------------------
# Options

option(CONVFELT_ENABLE_TESTS "Build/install tests and demos" OFF)


#------------------------------------------------------------
# Targets

# Convenience install target.

add_custom_target(
	convfelt.install
	COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
)

# Main source

add_subdirectory(src)

# Tests

if (CONVFELT_ENABLE_TESTS)
	message(STATUS "ConvFelt: Tests enabled")
	# Enable CTest target support (`add_test()`).
	enable_testing()
	add_subdirectory(tests)
else()
	message(STATUS "ConvFelt: Tests disabled")
endif ()


#------------------------------------------------------------
# CMake install package config

set(version_config_file "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config_file "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Version fetched from the top level project()
write_basic_package_version_file(
	${version_config_file}
	COMPATIBILITY SameMinorVersion
)

configure_package_config_file(
	cmake/Config.cmake.in
	${project_config_file}
	INSTALL_DESTINATION ${config_install_dir}
)

if (${PROJECT_NAME}_EXPORTED_TARGETS)
	install(
		EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
		DESTINATION ${config_install_dir}
		NAMESPACE ${PROJECT_NAME}::
		FILE ${PROJECT_NAME}Targets.cmake
	)
endif ()

install(
	FILES "${project_config_file}" "${version_config_file}"
	DESTINATION "${config_install_dir}"
)
