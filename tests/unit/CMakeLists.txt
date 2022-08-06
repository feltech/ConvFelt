#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.unit
	main.cpp
	test_ConvGrid.cpp
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
#	MKL::onemkl_blas_cublas
#	SyclBLAS # Broken for hipSYCL
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

catch_discover_tests(convfelt.test.unit
	PROPERTIES
	ENVIRONMENT
	LD_LIBRARY_PATH=${oneMKL_ROOT}/lib
)