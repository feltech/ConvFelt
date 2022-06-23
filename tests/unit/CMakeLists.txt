#------------------------------------------------------------
# Install testing libraries

# Install Catch2 testing library
CPMAddPackage("gh:catchorg/Catch2@2.13.8")
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)

# Install Trompeloeil mocking library
CPMAddPackage("gh:rollbear/trompeloeil@42")


#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.unit
	main.cpp
)

target_link_libraries(convfelt.test.unit
	PRIVATE
	project_options project_warnings
	convfelt)

target_link_system_libraries(convfelt.test.unit
	PRIVATE
	Catch2 trompeloeil)

catch_discover_tests(convfelt.test.unit)