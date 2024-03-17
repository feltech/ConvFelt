// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#include <string>

#include <convfelt/network/Network.hpp>

namespace convfelt::network
{

Network Network::from_yaml(std::string_view yaml)
{
	return {};
}

std::string Network::to_yaml() const
{
	return {};
}

void Network::initialise_with_random_weights() {}
} // namespace convfelt::network