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
)

target_compile_definitions(
	convfelt.test.unit
	PRIVATE
	CONVFELT_TEST_RESOURCE_DIR=\"${PROJECT_SOURCE_DIR}/tests/resources\"
)

catch_discover_tests(convfelt.test.unit)