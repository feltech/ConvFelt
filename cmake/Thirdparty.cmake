include_guard(GLOBAL)
include(GNUInstallDirs)

#---------------------------------------------------------------------------------------------------
# System dependencies

# Compiler-specific support.
#if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
#	message(FATAL_ERROR "OpenSYCL dependency requires clang compiler for all features ("
#		"esp. coroutines) to be available")
#elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)
#	add_compile_options(
#		-Wno-error=unknown-cuda-version
#	)
#	add_link_options(
#		-Wno-error=unknown-cuda-version
#		-Qunused-arguments
#	)
#endif ()


#---------------------------------------------------------------------------------------------------
# Add project_options CMake library

set(ProjectOptions_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor/project_options)
set(ProjectOptions_SRC_DIR ${ProjectOptions_DIR}/src CACHE FILEPATH "")
mark_as_advanced(ProjectOptions_SRC_DIR)
add_subdirectory(${ProjectOptions_DIR})
# Too many false positives, especially with templated code.
set(ENABLE_CPPCHECK_DEFAULT OFF)
# Interferes with OpenCL, so disable by default even for "developer mode".
set(ENABLE_SANITIZER_ADDRESS_DEFAULT OFF)
# TODO(DF): Add ASan-enabled test target using the following workarounds.
# Asan causes -1001 error in clGetPlatformIDs, which can be worked around by:
#   ASAN_OPTIONS=protect_shadow_gap=0
# ASan then reports leaks via LSan in NVIDIA OpenCL (v510), which can be suppressed by:
#   echo leak:libnvidia-opencl.so > lsan.supp
#   LSAN_OPTIONS=suppressions=lsan.supp
# With the above workarounds, OPT_ENABLE_SANITIZER_ADDRESS can be enabled.

# nvcc doesn't support -Werror... or -std=c++20
#set(WARNINGS_AS_ERRORS_DEFAULT OFF)
# Disable ccache as it confuses the hipSyCl compiler introspection (see syclcc-launcher)
set(ENABLE_CACHE_DEFAULT OFF)
# Disable for now, since -fcoroutines is not supported by Clang, but libstdc++ asserts on it.
#set(ENABLE_CLANG_TIDY_DEFAULT OFF)
# Disable for now, since false positive compile error, i.e.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95137#c42
#set(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR_DEFAULT OFF)
# Generate project_options and project_warnings INTERFACE targets.
include(${ProjectOptions_SRC_DIR}/DynamicProjectOptions.cmake)
dynamic_project_options()

#---------------------------------------------------------------------------------------------------
# Configure CMake defaults for friendlier package support.


# Allow dependencies with e.g. only Release CMake config files to work with Debug project builds.
#set(CMAKE_MAP_IMPORTED_CONFIG_RELEASE Release;RelWithDebInfo;Debug)
#set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo;Release;Debug)
#set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Debug;RelWithDebInfo;Release)

if (NOT DEFINED CMAKE_FIND_PACKAGE_PREFER_CONFIG)
	set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
endif ()
message(STATUS "CMAKE_FIND_PACKAGE_PREFER_CONFIG=${CMAKE_FIND_PACKAGE_PREFER_CONFIG}")

list(APPEND CMAKE_PREFIX_PATH ${convfelt_DEPENDENCY_INSTALL_CACHE_DIR})

#---------------------------------------------------------------------------------------------------
# External packages.

# SyCL support
find_package(AdaptiveCpp REQUIRED)
message(STATUS "ConvFelt: Using AdaptiveCpp from ${AdaptiveCpp_DIR}")

# MKL
if (NOT DEFINED HIPSYCL_TARGETS)
	# TODO(DF): remove once oneMKL migrates away from deprecated hipSYCL target.
	set(HIPSYCL_TARGETS "omp,cuda" CACHE STRING "Deprecated hipSYCL targets" FORCE)
	set(HIPSYCL_TARGETS "omp,cuda")
endif()
find_package(oneMKL CONFIG REQUIRED)
message(STATUS "ConvFelt: Using oneMKL from ${oneMKL_DIR} with targets: ${MKL_IMPORTED_TARGETS}")
target_compile_definitions(MKL::onemkl INTERFACE ENABLE_CUBLAS_BACKEND)

# SYCL-BLAS
# Broken for hipSYCL: https://github.com/codeplaysoftware/sycl-blas/issues/303

# Multi-dimensional array span.
find_package(mdspan REQUIRED)

# BLAS
find_package(Eigen3 REQUIRED)
# Work around Eigen<=3.4.0 C++20 incompatibility
target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_HAS_STD_RESULT_OF=0)

# Image reading/writing.
find_package(OpenImageIO REQUIRED)

# Range helpers..
find_package(range-v3 REQUIRED)

# Checking if boost::format works in kernels...
#find_package(Boost REQUIRED)

# Check if Embedded Template Library works in kernels
find_package(etl)

# TODO: cppcoro FindCoroutines bug - Coroutines_FOUND cache variable not read properly. Seems to
# be shadowed by a non-cache variable. My theory is it's the built-in *_FOUND variable
# initialized by CMake.
# Conan version is old and errors with:  TODO(DF): Is this still true?
# > andreasbuhr-cppcoro/cci.20210113: Invalid ID: andreasbuhr-cppcoro does not support clang
# > with libstdc++. Use libc++ instead.
# Must allow set(CACHE) to overwrite normal variables For cppcorro's FindCoroutines
# module. Wierd bug that won't reproduce on a minimal example.
cmake_policy(SET CMP0126 OLD)
find_package(cppcoro REQUIRED)
#if (NOT TARGET cppcoro::cppcoro)
#	convfelt_cpm_install_package(
#		NAME cppcoro
#		GITHUB_REPOSITORY karzhenkov/cppcoro
#		GIT_TAG 2f54023f5148ac13bb825c61fc568cb26a61978b
#	)
#endif ()

if (CONVFELT_ENABLE_TESTS)
	# Unit tests
	find_package(Catch2 REQUIRED)
	# Mocks
	find_package(trompeloeil REQUIRED)

	#	if (NOT TARGET Catch2::Catch2WithMain)
	#		convfelt_cpm_install_package(
	#			NAME Catch2
	#			GITHUB_REPOSITORY catchorg/Catch2
	#			GIT_TAG v3.4.0
	#		)
	#	endif ()
	#
	#	# Mocking
	#	if (NOT TARGET trompeloeil::trompeloeil)
	#		convfelt_cpm_install_package(
	#			NAME trompeloeil
	#			GITHUB_REPOSITORY rollbear/trompeloeil
	#			GIT_TAG v42
	#		)
	#	endif ()

	#	# BDD tests
	#	find_package(CucumberCpp QUIET)
	#	if (NOT TARGET CucumberCpp::cucumber-cpp-nomain)
	#		find_package(asio REQUIRED)
	#		find_package(nlohmann_json REQUIRED)
	#		find_package(yaml-cpp REQUIRED)
	#		find_package(fmt REQUIRED)
	#		find_package(Boost REQUIRED)
	#
	#		# File to patch CucumberCpp's CMake.
	#		file(
	#			WRITE ${PROJECT_BINARY_DIR}/_deps/cucumbercpp.injected.cmake
	#			"
	#		function (link_deps)
	#			find_package(asio REQUIRED)
	#			find_package(nlohmann_json REQUIRED)
	#			find_package(tclap REQUIRED)
	#			target_link_libraries(cucumber-cpp-internal PRIVATE nlohmann_json::nlohmann_json asio::asio)
	#			target_link_libraries(cucumber-cpp PRIVATE asio::asio nlohmann_json::nlohmann_json tclap::tclap)
	#			target_link_libraries(cucumber-cpp-nomain PRIVATE asio::asio nlohmann_json::nlohmann_json)
	#		endfunction()
	#		# Defer linking targets til they're all created.
	#		cmake_language(DEFER CALL link_deps())
	#		"
	#		)
	#
	#		set(
	#			_cucumber_cpp_cmake_options
	#			# Can't build as a separate shared lib because:
	#			# > undefined reference to `typeinfo for cucumber::internal::CukeEngine'
	#			-DBUILD_SHARED_LIBS=FALSE
	#			-DCUKE_ENABLE_GTEST=OFF
	#			-DCUKE_ENABLE_BOOST_TEST=OFF
	#			-DCUKE_ENABLE_QT=OFF
	#			-DCUKE_TESTS_UNIT=OFF
	#			-DCMAKE_PROJECT_Cucumber-Cpp_INCLUDE=${PROJECT_BINARY_DIR}/_deps/cucumbercpp.injected.cmake
	#		)
	#
	#		# Pass along any external override to Boost static vs. shared.
	#		if (DEFINED Boost_USE_STATIC_LIBS)
	#			set(_cucumber_cpp_cmake_options
	#				${_cucumber_cpp_cmake_options} -DCUKE_USE_STATIC_BOOST=${Boost_USE_STATIC_LIBS})
	#		endif ()
	#
	#		convfelt_cpm_install_package(
	#			NAME CucumberCpp
	#			GITHUB_REPOSITORY cucumber/cucumber-cpp
	#			GIT_TAG v0.7.0
	#			FIND_PACKAGE_OPTIONS
	#			PATH_SUFFIXES lib/cmake
	#			CMAKE_OPTIONS
	#			${_cucumber_cpp_cmake_options}
	#		)
	#	endif ()
	#
	#
	#	find_package(cucumber-cpp-runner QUIET)
	#	if (NOT TARGET cucumber-cpp-runner::cucumber-cpp-runner)
	#		find_package(yaml-cpp REQUIRED)
	#		find_package(asio REQUIRED)
	#		find_package(Boost REQUIRED)
	#
	#		convfelt_cpm_install_package(
	#			NAME cucumber-cpp-runner
	#			GITHUB_REPOSITORY feltech/cucumber-cpp-runner
	#			GIT_TAG d09e66cd
	#			CMAKE_OPTIONS
	#			# Can't discover without help since CucumberCpp installs its CMake config files to
	#			# non-standard location.
	#			-DCucumberCpp_DIR=${CucumberCpp_DIR}
	#		)
	#	endif ()
endif ()
