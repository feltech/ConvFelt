// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <string>
#include <string_view>

#include <convfelt/network/Network.hpp>

namespace convfelt::network
{

Network Network::from_yaml([[maybe_unused]] std::string_view yaml_)
{
	return {};
}

// TODO(DF): remove exclusion once implementation exists.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::string Network::to_yaml() const
{
	return {};
}

void Network::initialise_with_random_weights() {}
}  // namespace convfelt::network