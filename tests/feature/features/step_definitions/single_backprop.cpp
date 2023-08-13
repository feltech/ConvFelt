// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <fmt/printf.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <cucumber-cpp/autodetect.hpp>

#include <convfelt/InputImageGrid.hpp>

template <typename Sig>
struct signature;

template <typename R, typename Arg>
struct signature<R(Arg)>
{
	using rest = std::tuple<>;
};
template <typename R, typename Arg, typename... Args>
struct signature<R(Arg, Args...)>
{
	using rest = std::tuple<Args...>;
};

// template <typename Next, typename Current>
// concept ChainConstructible = requires(Current c) {
//	{
//		Next{std::move(c), std::declval < std::tuple_element<signature<Current>>()}
//	};
// };
//
// template <class Current>
// struct NetBuilder
//{
//	sycl::context ctx;
//	Current last;
//
//	template <ChainConstructible<Current> Next, typename... Args>
//	constexpr Next then(Args &... args) const &&
//	{
//		return Builder{Next{std::move(last), std::move(ctx), std::forward<Args>(args)...}};
//	}
// };
//
GIVEN("^image '(.*)'$")
{
	REGEX_PARAM(std::string, file_name);
	[[maybe_unused]] auto thing = convfelt::InputImageGrid::make_from_file(file_name);
};
GIVEN(R"(^a model defined by the following YAML$)")
{
	REGEX_PARAM(std::string, model_spec);

	auto spec = YAML::Load(model_spec);

	auto filter_width = spec[0]["size"]["width"].as<std::size_t>();

	//		ImageGrid template_image{file_path};
	//
	//		auto classifier = convfelt::network::Builder{ctx, std::move(template_image)}
	//							  .then<Conv>(Vec2i{3, 3}, Vec2i{2, 2})
	//							  .then<Relu>()
	//							  .then<FullyConnected>()
	//							  .then<SoftMax>()
	//							  .last;
	//
	//		classifier.input().load_image(file_path);
	//		classifier.inference();
	//		classifier.backprop();
};
GIVEN("model is initialised with random weights"){

};

WHEN("inference is performed on the image"){};
THEN(R"(the result is in \[(\d+), (\d+)\] for each class)"){};
THEN(R"(the sum across classes is (\d+))"){};
THEN("repeated inferences give the same result"){};

GIVEN(R"(the correct class for the image is (\d+))"){};
WHEN("backprop is used to update the weights"){};
THEN("^he value for class 1 is greater after backprop than before$"){};
