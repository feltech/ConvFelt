// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <GUnit/GSteps.h>
#include <GUnit/GTest.h>

GSTEPS("*")
{
	Given("image '{file}'") = [&]([[maybe_unused]] std::string const & file_name) {

	};
	Given(R"(a '{model_spec}' model)") = [&]([[maybe_unused]] std::string const & model_spec)
	{
		Given("model is initialised with random weights") = [&] {

		};
	};

	When("inference is performed on the image") = [&] {};
	Then(R"(the result is in ['{start}', '{end}'] for each class)") =
		[&]([[maybe_unused]] float const start, [[maybe_unused]] float const end) {};
	Then("the sum across classes is {sum}") = [&]([[maybe_unused]] float const sum) {};
	Then("repeated inferences give the same result") = [&] {};

	Given("the correct class for the image is {correct_class}") =
		[&]([[maybe_unused]] std::size_t const correct_class) {};
	When("backprop is used to update the weights") = [&] {};
	Then("the value for class 1 is greater after backprop than before") = [&] {};
}
