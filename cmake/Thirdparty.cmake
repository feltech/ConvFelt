#------------------------------------------------------------
# System dependencies

# Compiler-specific support.
if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
	message(FATAL_ERROR "OpenSYCL dependency requires clang compiler for all features ("
		"esp. coroutines) to be available")
elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)
	add_compile_options(
		-Wno-error=unknown-cuda-version
	)
	add_link_options(
		-Wno-error=unknown-cuda-version
		-Qunused-arguments
	)
endif ()

#------------------------------------------------------------
# SyCL support

if (NOT HIPSYCL_TARGETS)
	set(HIPSYCL_TARGETS omp;cuda.integrated-multipass:sm_70 CACHE STRING
		"hipSycl compilation flow targets")
endif ()

find_package(hipSYCL REQUIRED CONFIG)

message(STATUS "ConvFelt: Using hipSyCL from ${hipSYCL_DIR}")


#------------------------------------------------------------
# Install CPM package manager

set(CPM_DOWNLOAD_VERSION 0.34.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
	message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
	file(DOWNLOAD
		https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
		${CPM_DOWNLOAD_LOCATION}
	)
endif ()

include(${CPM_DOWNLOAD_LOCATION})
set(CONVFELT_DEPENDENCIES_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dependencies" CACHE PATH
	"Install tree for build dependencies")
list(APPEND CMAKE_PREFIX_PATH ${CONVFELT_DEPENDENCIES_INSTALL_PREFIX})

#------------------------------------------------------------
# Install Conan package manager CMake helpers

# conan.cmake uses build tree, and can't be configured otherwise, annoyingly.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_BINARY_DIR})

if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
	message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
	file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
		"${CMAKE_BINARY_DIR}/conan.cmake"
		TLS_VERIFY ON)
endif ()

include(${CMAKE_BINARY_DIR}/conan.cmake)

#------------------------------------------------------------
# SYCL-BLAS
# Broken for hipSYCL: https://github.com/codeplaysoftware/sycl-blas/issues/303

#CPMAddPackage(
#	NAME SyclBLAS
#	GIT_TAG master
#	GIT_REPOSITORY https://github.com/codeplaysoftware/sycl-blas.git
#	OPTIONS
#	"BLAS_BUILD_SAMPLES OFF"
#	"BLAS_ENABLE_TESTING OFF"
#	"BLAS_ENABLE_BENCHMARK OFF"
#	"SYCL_COMPILER hipsycl"
#	"BUILD_SHARED_LIBS ON"
#	"BLAS_ENABLE_CONST_INPUT ON"
#	"ENABLE_EXPRESSION_TESTS OFF"
#	"BLAS_VERIFY_BENCHMARK OFF"
#	"BLAS_ENABLE_EXTENSIONS ON"
#)
#find_package(SyclBLAS REQUIRED)

# Compilation fails because of a missing function signature. Disabling BLAS_ENABLE_CONST_INPUT
# should skip that bit of code, but it doesn't because of a bug - #ifdef is used rather than
# checking the value.
#get_directory_property(defs ${SyclBLAS_SOURCE_DIR}/src/policy COMPILE_DEFINITIONS)
#list(FILTER defs EXCLUDE REGEX BLAS_ENABLE_CONST_INPUT)
#set_property(DIRECTORY ${SyclBLAS_SOURCE_DIR}/src/policy PROPERTY COMPILE_DEFINITIONS ${defs})


#------------------------------------------------------------
# MKL
# APT package doesn't work with libc++, must build.
find_package(oneMKL CONFIG REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
target_compile_definitions(MKL::onemkl INTERFACE ENABLE_CUBLAS_BACKEND)

#------------------------------------------------------------
# Add project_options CMake library

CPMAddPackage("gh:cpp-best-practices/project_options@0.22.4")
include(${project_options_SOURCE_DIR}/src/DynamicProjectOptions.cmake)
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
set(ENABLE_CLANG_TIDY_DEFAULT OFF)
# Disable for now, since false positive compile error, i.e.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95137#c42
set(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR_DEFAULT OFF)


# Generate project_options and project_warnings INTERFACE targets.
dynamic_project_options()

function(convfelt_cpm_install_package package_name gh_repo git_tag)
	CPMAddPackage(
		NAME ${package_name}
		SYSTEM YES
		EXCLUDE_FROM_ALL YES
		DOWNLOAD_ONLY YES
		GIT_TAG ${git_tag}
		GIT_REPOSITORY https://github.com/${gh_repo}.git
	)
	find_package(${package_name} CONFIG QUIET)
	if (NOT DEFINED ${package_name}_CONFIG)
		execute_process(
			COMMAND ${CMAKE_COMMAND}
			-S ${${package_name}_SOURCE_DIR} -B ${${package_name}_BINARY_DIR}
			-DCMAKE_BUILD_TYPE=RelWithDebInfo
			COMMAND_ERROR_IS_FATAL ANY
		)
		execute_process(
			COMMAND ${CMAKE_COMMAND}
			--build ${${package_name}_BINARY_DIR} --config RelWithDebInfo --parallel
			COMMAND_ERROR_IS_FATAL ANY
		)
		execute_process(
			COMMAND ${CMAKE_COMMAND}
			--install ${${package_name}_BINARY_DIR} --prefix ${CONVFELT_DEPENDENCIES_INSTALL_PREFIX}
			COMMAND_ERROR_IS_FATAL ANY
		)
	endif ()
	find_package(${package_name} CONFIG REQUIRED)
endfunction()

#------------------------------------------------------------
# cppcoro C++20 coroutines library
# Use CPM rather than Conan, since the (current) Conan recipe forces `-fcoroutines-ts`, which is
# removed in clang-17.
convfelt_cpm_install_package(cppcoro andreasbuhr/cppcoro main)

#------------------------------------------------------------
# Testing libraries

if (CONVFELT_ENABLE_TESTS)

	#------------------------------------------------------------
	# Catch2 testing library

	convfelt_cpm_install_package(Catch2 catchorg/Catch2 v3.4.0)

	#------------------------------------------------------------
	# Trompeloeil mocking library

	convfelt_cpm_install_package(trompeloeil rollbear/trompeloeil v42)
endif ()

#------------------------------------------------------------
# Load dependencies via Conan package manager

conan_cmake_configure(
	REQUIRES

	# Pin to boost version such that OpenImageIO supports boost::filesystem without undefined
	# references
	boost/1.77.0

	# String formatting. Must override dependency of openimageio or get ambiguous calls to
	# `std::signbit` et al.
	fmt/7.1.3

	# Eigen linear algebra library
	eigen/3.4.0

	# OpenImageIO image loading/processing library
	openimageio/2.4.7.1
	# Override to >=v3 so that Imath is used, rather than a fallback (OIIO_USING_IMATH), which works
	# around "error: definition of type 'half' conflicts with typedef of the same name" when CUDA
	# (via OpenSYCL) is also included.
	openexr/3.1.5

	# range-v3 (whilst waiting for std::range) library
	range-v3/0.12.0

	GENERATORS cmake_find_package

	OPTIONS
	openimageio:shared=True
	openimageio:fPIC=True
	openimageio:with_libjpeg=libjpeg
	openimageio:with_libpng=True
	openimageio:with_freetype=False
	openimageio:with_hdf5=False
	openimageio:with_opencolorio=False
	openimageio:with_opencv=False
	openimageio:with_tbb=False
	openimageio:with_dicom=False # Heavy dependency disabled by default
	openimageio:with_ffmpeg=False
	openimageio:with_giflib=False
	openimageio:with_libheif=False
	openimageio:with_raw=False # libraw is available under CDDL-1.0 or LGPL-2.1 for this reason it is disabled by default
	openimageio:with_openjpeg=False
	openimageio:with_openvdb=False # FIXME broken on M1
	openimageio:with_ptex=False
	openimageio:with_libwebp=False
)

if (NOT DEFINED CONVFELT_CONAN_PROFILE)
	set(CONVFELT_CONAN_PROFILE default)
endif ()
if (NOT DEFINED CONVFELT_CONAN_REMOTE)
	set(CONVFELT_CONAN_REMOTE conancenter)
endif ()

conan_cmake_install(
	PATH_OR_REFERENCE ${CMAKE_CURRENT_BINARY_DIR}
	BUILD missing
	REMOTE ${CONVFELT_CONAN_REMOTE}
	PROFILE ${CONVFELT_CONAN_PROFILE}
	# Disable error for warning triggered by boost mpi build.
	ENV CPPFLAGS=-Wno-error=enum-constexpr-conversion
)

find_package(Eigen3 REQUIRED)
# Work around Eigen<=3.4.0 C++20 incompatibility
target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_HAS_STD_RESULT_OF=0)
find_package(OpenImageIO REQUIRED)
#add_library(cppcoro::cppcoro ALIAS cppcoro)
find_package(range-v3 REQUIRED)
