include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(STATUS "Injecting install logic into ${PROJECT_NAME}")
get_filename_component(inject_script_dir ${CMAKE_PROJECT_GUnit_INCLUDE} DIRECTORY)

# All the static libs (all 5 or so) will be combined into a shared lib, and we want to expose their
# symbols.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(configure_install_target)

	# Prevent build-only dependencies from being installed.
	set_property(DIRECTORY ${PROJECT_SOURCE_DIR}/libs/gherkin-cpp/libs/fmem PROPERTY EXCLUDE_FROM_ALL TRUE)
	set_property(DIRECTORY ${PROJECT_SOURCE_DIR}/libs/gherkin-cpp/libs/gherkin-c PROPERTY EXCLUDE_FROM_ALL TRUE)

	# We're going to compile everything into a shared lib with WHOLE_ARCHIVE option, so avoid
	# conflicts by blanking out source gunit target's interface.
	set_target_properties(
		gunit
		PROPERTIES
		INTERFACE_LINK_LIBRARIES ""
	)
	# Create wrapper shared library to bundle GUnit and all its binary dependencies.
	#
	# Much much simpler than trying to shim all the bad CMake of the various dependencies to create
	# lots of separate installable targets.
	add_library(gunit_so SHARED)
	target_link_libraries(gunit_so PRIVATE gunit)
	set_target_properties(
		gunit_so
		PROPERTIES
		EXPORT_NAME gunit
		OUTPUT_NAME gunit
	)
	target_include_directories(
		gunit_so INTERFACE
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		# Frustratingly, GUnit expects gherkin-cpp headers to be in the top-level of search path.
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gherkin-cpp>
	)
	# Link GUnit's (static) dependencies into the wrapper shared library, ensuring that all their
	# symbols are exported (WHOLE_ARCHIVE).
	target_link_libraries(
		gunit_so PRIVATE
		$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gtest_main gmock gherkin-cpp>>
	)
	file(GLOB_RECURSE gunit_header_files LIST_DIRECTORIES false CONFIGURE_DEPENDS
		"${PROJECT_SOURCE_DIR}/include/*")
	target_sources(
		gunit_so
		PUBLIC FILE_SET HEADERS
		BASE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/libs/json/single_include/nlohmann"
		FILES ${gunit_header_files} libs/json/single_include/nlohmann/json.hpp)

	file(
		WRITE ${PROJECT_BINARY_DIR}/GUnitConfig.cmake.in [=[
set(@PROJECT_NAME@_VERSION @PROJECT_VERSION@)

@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(GTest REQUIRED)
include ("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")
]=]
	)

	configure_package_config_file(
		${PROJECT_BINARY_DIR}/GUnitConfig.cmake.in GUnitConfig.cmake
		INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
	)
	install(
		FILES ${CMAKE_CURRENT_BINARY_DIR}/GUnitConfig.cmake
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
	)
	install(
		TARGETS gunit_so
		EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
		FILE_SET HEADERS DESTINATION include
	)
	install(
		EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
		NAMESPACE ${PROJECT_NAME}::
		FILE ${PROJECT_NAME}Targets.cmake
	)
endmacro()

cmake_language(
	DEFER CALL configure_install_target
)