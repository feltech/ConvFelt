include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(STATUS "Injecting install logic into ${PROJECT_NAME}")

# All the static libs (all 4 or so) will be combined into a shared lib, and we want to expose their
# symbols.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(configure_gherkincpp_install_target)
	# Prevent build-only dependencies from being fully installed.  I.e. we don't want to bundle
	# their headers. We'll bundle the libraries themselves together with gherkin-cpp, below.
	set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/libs/fmem PROPERTY EXCLUDE_FROM_ALL TRUE)
	set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/libs/gherkin-c PROPERTY EXCLUDE_FROM_ALL TRUE)
	# Don't bother building unit tests.
	set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/test PROPERTY EXCLUDE_FROM_ALL TRUE)
	# Blank include path for install target. Project has set them to absolute paths, which doesn't
	# work for install targets. Include path for install target added to top-level gherkin-cpp
	# target, below.
	foreach (target_name gherkin-cpp fmem)
		# ^ `gherkin` target is already configured properly with BUILD_INTERFACE guard.
		get_target_property(
			${target_name}_INTERFACE_INCLUDE_DIRECTORIES
			${target_name}
			INTERFACE_INCLUDE_DIRECTORIES
		)
		set_target_properties(
			${target_name}
			PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES ""
		)
		target_include_directories(
			${target_name}
			PUBLIC
			$<BUILD_INTERFACE:${${target_name}_INTERFACE_INCLUDE_DIRECTORIES}>
		)
	endforeach ()
	# Configure include path for top-level gherkin-cpp target.
	target_include_directories(
		gherkin-cpp
		PUBLIC
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		# Frustratingly, GUnit expects gherkin-cpp headers to be in the top-level of search path.
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gherkin-cpp>
	)
	# Do the package config file dance.
	file(
		WRITE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in [=[
set(@PROJECT_NAME@_VERSION @PROJECT_VERSION@)

@PACKAGE_INIT@

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
		TARGETS gherkin-cpp fmem gherkin
		EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
	)
	install(
		EXPORT ${PROJECT_NAME}_EXPORTED_TARGETS
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
		NAMESPACE ${PROJECT_NAME}::
		FILE ${PROJECT_NAME}Targets.cmake
	)
endmacro()

cmake_language(
	DEFER CALL configure_gherkincpp_install_target
)