include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(STATUS "Injecting install logic into ${PROJECT_NAME}")

# All the static libs (all 4 or so) will be combined into a shared lib, and we want to expose their
# symbols.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(configure_gunit_install_target)
	get_target_property(gunit_INTERFACE_LINK_LIBRARIES gunit INTERFACE_LINK_LIBRARIES)
	get_target_property(gunit_INTERFACE_INCLUDE_DIRECTORIES gunit INTERFACE_INCLUDE_DIRECTORIES)
	# We're going to compile everything into a shared lib with WHOLE_ARCHIVE option, so avoid
	# conflicts by blanking out source gunit target's install interface.
	set_target_properties(
		gunit
		PROPERTIES
		INTERFACE_LINK_LIBRARIES ""
		INTERFACE_INCLUDE_DIRECTORIES ""
	)
	target_link_libraries(
		gunit
		INTERFACE
		$<BUILD_INTERFACE:${gunit_INTERFACE_LINK_LIBRARIES}>
	)
	target_include_directories(
		gunit
		INTERFACE
		$<BUILD_INTERFACE:${gunit_INTERFACE_INCLUDE_DIRECTORIES}>
	)
	# Create wrapper shared library to bundle GUnit and all its binary dependencies.
	#
	# Much much simpler than trying to shim all the bad CMake of the various dependencies to create
	# lots of separate installable static library targets with dependencies between them.
	add_library(gunit_so SHARED)
	set_target_properties(
		gunit_so
		PROPERTIES
		EXPORT_NAME gunit
		OUTPUT_NAME gunit
	)
	target_link_libraries(
		gunit_so 
		PUBLIC
		# Link GUnit's (static) dependencies into the wrapper shared library, ensuring that all
		# their symbols are exported (WHOLE_ARCHIVE).
		$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,gtest_main gmock gherkin-cpp>>
		# For the install interface, "link" library dependencies, but only header paths, compile
		# flags, etc, not the (static) libraries themselves since they're baked into the shared lib
		# (hence COMPILE_ONLY).
		$<INSTALL_INTERFACE:$<COMPILE_ONLY:gherkin-cpp>>
		$<INSTALL_INTERFACE:$<COMPILE_ONLY:gtest>>
	)
	target_include_directories(
		gunit_so
		PUBLIC
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	)

	# Install GUnit headers using new-style FILE_SET feature.

	file(GLOB_RECURSE gunit_header_files LIST_DIRECTORIES false CONFIGURE_DEPENDS
		"${PROJECT_SOURCE_DIR}/include/*")
	target_sources(
		gunit_so
		PUBLIC FILE_SET HEADERS
		BASE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/libs/json/single_include/nlohmann"
		# TODO(DF): Not happy that json.hpp is in the top-level `include` directory.
		FILES ${gunit_header_files} libs/json/single_include/nlohmann/json.hpp)

	# Do the package config file dance.
	file(
		WRITE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in [=[
set(@PROJECT_NAME@_VERSION @PROJECT_VERSION@)

@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(GTest REQUIRED)
find_dependency(gherkin-cpp REQUIRED)

include ("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")
]=]
	)
	configure_package_config_file(
		${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in ${PROJECT_NAME}Config.cmake
		INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
	)
	install(
		FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
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
	DEFER CALL configure_gunit_install_target
)