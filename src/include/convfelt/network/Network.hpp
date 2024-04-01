// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once

#include <string_view>

namespace convfelt::network
{

class Network
{
public:
	[[nodiscard]] static Network from_yaml(std::string_view yaml_);
	void initialise_with_random_weights();
	[[nodiscard]] std::string to_yaml() const;
};

}  // namespace convfelt::network
