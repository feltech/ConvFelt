// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <memory>

#include <boost/test/unit_test.hpp>
#include <cucumber-cpp/autodetect.hpp>
#include <fmt/printf.h>
#include <yaml-cpp/yaml.h>

#include <boost/test/tools/interface.hpp>
#include <convfelt/InputImageGrid.hpp>
#include <convfelt/network/Network.hpp>

template <typename Sig>
struct signature;

template <typename R, typename Arg>
struct signature<R(Arg)>
{
	using rest = std::tuple<Arg>;
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

using cucumber::ScenarioScope;

namespace
{
struct Context
{
	std::unique_ptr<convfelt::network::Network> net;
	YAML::Node input_spec;
	YAML::Node output_spec;
};
}  // namespace

GIVEN("^image '(.*)'$")
{
	REGEX_PARAM(std::string, file_name);
	[[maybe_unused]] auto thing = convfelt::InputImageGrid::make_from_file(file_name);
};

GIVEN(R"(^a model defined by the following YAML$)")
{
	REGEX_PARAM(std::string, model_spec);

	//	auto spec = YAML::Load(model_spec);
	//	auto filter_width = spec[0]["size"]["width"].as<std::size_t>();

	convfelt::network::Network net = convfelt::network::Network::from_yaml(model_spec);

	ScenarioScope<Context> ctx;
	ctx->input_spec = YAML::Load(model_spec);
	ctx->net = std::make_unique<convfelt::network::Network>(std::move(net));
};

GIVEN("model is initialised with random weights")
{
	ScenarioScope<Context> ctx;
	ctx->net->initialise_with_random_weights();
};

WHEN("YAML is generated from the network structure")
{
	ScenarioScope<Context> ctx;
	ctx->output_spec = YAML::Load(ctx->net->to_yaml());
};
THEN("input YAML matches output YAML")
{
	ScenarioScope<Context> ctx;
	try
	{
		BOOST_TEST(ctx->input_spec == ctx->output_spec);
	}
	catch (std::exception const & exc)
	{
		fmt::print(exc.what());
		throw;
	}
}

WHEN("inference is performed on the image$"){};
THEN(R"(the result is in \[(\d+), (\d+)\] for each class)"){};
THEN(R"(the sum across classes is (\d+))"){};

WHEN(R"(inference is performed on the image (\d+) times)"){};
THEN("the inferred values haven't changed"){};

GIVEN(R"(the correct class for the image is (\d+))"){};
WHEN("backprop is used to update the weights"){};
THEN(R"(^the value for class (\d+) has increased)"){};
