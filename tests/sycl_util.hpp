// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <format>
#include <iterator>
#include <stdexcept>
#include <sycl/sycl.hpp>

inline void async_handler(sycl::exception_list const & error_list_)
{
	if (!error_list_.empty())
	{
		std::string errors;
		for (std::size_t idx = 0; idx < error_list_.size(); ++idx)
		{
			try
			{
				if (error_list_[idx])
				{
					std::rethrow_exception(error_list_[idx]);
				}
			}
			catch (std::exception & e)
			{
				std::format_to(std::back_inserter(errors), "{}: {}\n", idx, e.what());
			}
			catch (...)
			{
				std::format_to(
					std::back_inserter(errors), "{}: <unknown non-std::exception>\n", idx);
			}
		}
		throw std::runtime_error{errors};
	}
}
