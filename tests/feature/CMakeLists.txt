#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.feature
	main.cpp
	features/step_definitions/single_backprop.cpp
)
target_link_libraries(
	convfelt.test.feature
	PRIVATE
	CucumberCpp::cucumber-cpp-nomain
	Boost::program_options
	fmt::fmt
	convfelt
)

add_test(
	NAME convfelt.test.feature
#	COMMAND convfelt.test.feature
	COMMAND $<TARGET_FILE:convfelt.test.feature> --options "--strict"
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/features
)