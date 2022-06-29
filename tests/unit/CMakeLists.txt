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
	Catch2 trompeloeil)

catch_discover_tests(convfelt.test.unit)