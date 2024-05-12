#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.unit
	third_party.cpp
	test_felt2.cpp
)
add_test(convfelt.test.unit convfelt.test.unit)

target_link_libraries(
	convfelt.test.unit
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
	trompeloeil::trompeloeil
	OpenImageIO::OpenImageIO
	cppcoro::cppcoro
	range-v3::range-v3
	MKL::onemkl
	std::mdspan
)

add_sycl_to_target(TARGET convfelt.test.unit)

target_compile_definitions(
	convfelt.test.unit
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
	# Disable fatal assertions.
	FELT2_DEBUG_NONFATAL
)