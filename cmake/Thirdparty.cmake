include_guard(GLOBAL)
include(GNUInstallDirs)

#---------------------------------------------------------------------------------------------------
# Add project_options CMake library

set(ProjectOptions_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor/project_options)
set(ProjectOptions_SRC_DIR ${ProjectOptions_DIR}/src CACHE FILEPATH "")
mark_as_advanced(ProjectOptions_SRC_DIR)
add_subdirectory(${ProjectOptions_DIR})

# Too many false positives, especially with templated code.
set(ENABLE_CPPCHECK_DEFAULT OFF)

# SYCL JIT CUDA kernel fail: cuda_hardware_manager: Could not obtain number of devices (error code = CUDA:2)
set(ENABLE_SANITIZER_ADDRESS_DEFAULT OFF)
# OpenCL ASan causes -1001 error in clGetPlatformIDs, which can be worked around by:
#   ASAN_OPTIONS=protect_shadow_gap=0
# ASan then reports leaks via LSan in NVIDIA OpenCL (v510), which can be suppressed by:
#   echo leak:libnvidia-opencl.so > lsan.supp
#   LSAN_OPTIONS=suppressions=lsan.supp

# SYCL JIT CUDA kernel fail: Unresolved extern function '__ubsan_handle_pointer_overflow'.
set(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR_DEFAULT OFF)

# Generate project_options and project_warnings INTERFACE targets.
include(${ProjectOptions_SRC_DIR}/DynamicProjectOptions.cmake)
dynamic_project_options()

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
#target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_HAS_STD_RESULT_OF=0)

# Image reading/writing.
find_package(OpenImageIO REQUIRED)

# Range helpers..
find_package(range-v3 REQUIRED)

# Checking if boost::format works in kernels...
#find_package(Boost REQUIRED)

# Embedded Template Library - useful for kernels where dynamic allocation is unavailable, e.g. strings.
find_package(etl)

# Corouties library.
find_package(cppcoro REQUIRED)

if (CONVFELT_ENABLE_TESTS)
	# Unit tests
	find_package(Catch2 REQUIRED)
	# Mocks
	find_package(trompeloeil REQUIRED)
endif ()
