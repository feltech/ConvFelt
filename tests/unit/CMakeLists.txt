#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.unit
	felt2/components/test_sycl.cpp
	felt2/test_typedefs.cpp
	test_ConvGrid.cpp
	test_FilterSizeHelper.cpp
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
	std::mdspan
)

add_sycl_to_target(TARGET convfelt.test.unit)

target_compile_definitions(
	convfelt.test.unit
	PRIVATE
	# Disable fatal assertions. For testing debug assert logging.
	FELT2_DEBUG_NONFATAL
)