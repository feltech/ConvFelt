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
# Add project_options CMake library v0.21.0

CPMAddPackage("gh:cpp-best-practices/project_options@0.21.0")
include(${project_options_SOURCE_DIR}/src/DynamicProjectOptions.cmake)
