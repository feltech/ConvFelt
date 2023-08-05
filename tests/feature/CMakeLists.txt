#------------------------------------------------------------
# Test executable target

add_executable(
	convfelt.test.feature
	features/step_definitions/single_backprop.cpp
)
target_link_libraries(convfelt.test.feature PRIVATE GUnit::gunit)