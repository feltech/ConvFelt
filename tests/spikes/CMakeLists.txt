#-----------------------------------------------------------------------------------------------------------------------
# Read from OIIO in a coroutine then paste into a convfelt grid.

add_executable(
	convfelt.test.spikes.oiio_to_grid_via_coroutine
	oiio_to_grid_via_coroutine.cpp
)
add_test(convfelt.test.spikes.oiio_to_grid_via_coroutine convfelt.test.spikes.oiio_to_grid_via_coroutine)

target_link_libraries(
	convfelt.test.spikes.oiio_to_grid_via_coroutine
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
	OpenImageIO::OpenImageIO
	cppcoro::cppcoro
)

target_compile_definitions(
	convfelt.test.spikes.oiio_to_grid_via_coroutine
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
)

#-----------------------------------------------------------------------------------------------------------------------
# Read from OIIO then spread out into convolution filter input regions.

add_executable(
	convfelt.test.spikes.oiio_to_filter_inputs
	oiio_to_filter_inputs.cpp
)
add_test(convfelt.test.spikes.oiio_to_filter_inputs convfelt.test.spikes.oiio_to_filter_inputs)

target_link_libraries(
	convfelt.test.spikes.oiio_to_filter_inputs
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
	OpenImageIO::OpenImageIO
)

target_compile_definitions(
	convfelt.test.spikes.oiio_to_filter_inputs
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
)

#-----------------------------------------------------------------------------------------------------------------------
# Hello World SYCL

add_executable(
	convfelt.test.spikes.basic_sycl
	basic_sycl.cpp
)
add_test(convfelt.test.spikes.basic_sycl convfelt.test.spikes.basic_sycl)

target_link_libraries(
	convfelt.test.spikes.basic_sycl
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET convfelt.test.spikes.basic_sycl)

#-----------------------------------------------------------------------------------------------------------------------
# Hello World oneMKL

add_executable(
	convfelt.test.spikes.basic_onemkl
	basic_onemkl.cpp
)
add_test(convfelt.test.spikes.basic_onemkl convfelt.test.spikes.basic_onemkl)

target_link_libraries(
	convfelt.test.spikes.basic_onemkl
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
	MKL::onemkl
)
add_sycl_to_target(TARGET convfelt.test.spikes.basic_onemkl)

#-----------------------------------------------------------------------------------------------------------------------
# Experiments applying convolution filter kernel to an image.

add_executable(
	convfelt.test.spikes.applying_convolution_filter
	applying_convolution_filter.cpp
)
add_test(convfelt.test.spikes.applying_convolution_filter convfelt.test.spikes.applying_convolution_filter)

target_link_libraries(
	convfelt.test.spikes.applying_convolution_filter
	PRIVATE
	project_options
	project_warnings
	convfelt
	# 3rd party
	Catch2::Catch2WithMain
	OpenImageIO::OpenImageIO
	MKL::onemkl
)
add_sycl_to_target(TARGET convfelt.test.spikes.applying_convolution_filter)

target_compile_definitions(
	convfelt.test.spikes.applying_convolution_filter
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
)
