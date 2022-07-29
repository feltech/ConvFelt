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


#------------------------------------------------------------
# Install Conan package manager CMake helpers

# conan.cmake uses build tree, and can't be configured otherwise, annoyingly.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
	message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
	file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
		"${CMAKE_BINARY_DIR}/conan.cmake"
		TLS_VERIFY ON)
endif ()

include(${CMAKE_BINARY_DIR}/conan.cmake)


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

# Disable for now, since -fcoroutines is not supported by Clang, but libstdc++ asserts on it.
set(ENABLE_CLANG_TIDY_DEFAULT OFF)
# Disable for now, since false positive compile error, i.e.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95137#c42
set(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR_DEFAULT OFF)



# Generate project_options and project_warnings INTERFACE targets.
dynamic_project_options()


#------------------------------------------------------------
# ViennaCL

CPMAddPackage(
	NAME ViennaCL
	GIT_TAG release-1.7.1
	GIT_REPOSITORY https://github.com/viennacl/viennacl-dev
	OPTIONS "BUILD_TESTING OFF" "BUILD_EXAMPLES OFF" "ENABLE_UBLAS OFF" "ENABLE_OPENCL ON"
)

if (ViennaCL_ADDED)
	# Disable clang-tidy for library target.
	set_target_properties(viennacl PROPERTIES CXX_CLANG_TIDY "")
	# Create include directory to gather relevant headers - to avoid pollution of adding whole
	# source tree to include path.
	file(MAKE_DIRECTORY ${ViennaCL_BINARY_DIR}/include)
	# Symlink relevant header directories
	file(CREATE_LINK
		${ViennaCL_SOURCE_DIR}/viennacl
		${ViennaCL_BINARY_DIR}/include/viennacl
		COPY_ON_ERROR SYMBOLIC)
	# Create library target
	add_library(ViennaCL::viennacl INTERFACE IMPORTED)
	# Add headers to library target.
	target_include_directories(ViennaCL::viennacl INTERFACE ${ViennaCL_BINARY_DIR}/include)
	# Link library target to source target.
	target_link_libraries(ViennaCL::viennacl INTERFACE viennacl)
endif ()


#------------------------------------------------------------
# Load dependencies via Conan package manager

conan_cmake_configure(
	REQUIRES

	# Eigen linear algebra library
	eigen/3.3.9

	# OpenImageIO image loading/processing library
	openimageio/2.2.18.0

	# cppcoro coroutines library
	andreasbuhr-cppcoro/cci.20210113

	# range-v3 (whilst waiting for std::range) library
	range-v3/0.12.0

	GENERATORS cmake_find_package

	OPTIONS
	openimageio:shared=False
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

conan_cmake_autodetect(conan_settings)

conan_cmake_install(
	PATH_OR_REFERENCE ${CMAKE_CURRENT_BINARY_DIR}
	BUILD missing
	REMOTE conancenter
	SETTINGS ${conan_settings}
)

find_package(Eigen3 REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(cppcoro REQUIRED)
find_package(range-v3 REQUIRED)


#------------------------------------------------------------
# (My) Felt library

CPMAddPackage(gh:feltech/Felt@3.2)


#------------------------------------------------------------
# Testing libraries

if (CONVFELT_ENABLE_TESTS)
	#------------------------------------------------------------
	# Catch2 testing library
	CPMAddPackage("gh:catchorg/Catch2@2.13.8")
	include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)

	#------------------------------------------------------------
	# Trompeloeil mocking library
	CPMAddPackage("gh:rollbear/trompeloeil@42")
endif ()
