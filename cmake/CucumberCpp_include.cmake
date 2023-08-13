include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(STATUS "Injecting install logic into ${PROJECT_NAME}")

# All the static libs (all 4 or so) will be combined into a shared lib, and we want to expose their
# symbols.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(configure_cucumbercpp_install_target)
	install(
		EXPORT      CucumberCpp
		NAMESPACE   CucumberCpp::
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CucumberCpp
		FILE        CucumberCppConfig.cmake
	)
endmacro()

cmake_language(
	DEFER CALL configure_cucumbercpp_install_target
)