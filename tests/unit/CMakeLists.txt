#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.unit
	main.cpp
	third_party.cpp
)

target_link_libraries(
	convfelt.test.unit
	PRIVATE
	project_options
	project_warnings
	convfelt
)

target_link_system_libraries(
	convfelt.test.unit
	PRIVATE
	Catch2 trompeloeil
	OpenImageIO::OpenImageIO
	cppcoro::cppcoro
	range-v3::range-v3
	MKL::onemkl
	hipSYCL::hipSYCL-rt
)


add_sycl_to_target(TARGET convfelt.test.unit)

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
	target_compile_options(
		convfelt.test.unit
		PRIVATE
		-fconcepts-diagnostics-depth=10
	)
endif ()

target_compile_definitions(
	convfelt.test.unit
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
)

target_link_options(
	convfelt.test.unit
	PRIVATE
	# oneMKL does not have RUNPATH set to locate sibling libraries, so use RPATH behaviour so that
	# our link search path is transient. In the build tree, CMake will set the RPATH of our binary
	# to include the link_directories of the oneML library (see ThirdParty.cmake). This is useful
	# for unit tests, since they run in the build tree.
	-Wl,--disable-new-dtags
)