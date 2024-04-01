// Copyright 2023 David Feltell
// SPDX-License-Identifier: MIT
#pragma once
#include <cassert>
#ifdef __CUDA_ARCH__
// This doesn't expand for device functions, yet is used by `assert`, so set a default.
#define __PRETTY_FUNCTION__ "<unknown>"
extern "C"
{
	// Add missing __host__ version. Unused, only required for compilation.
	// See __clang_cuda_runtime_wrapper.h
	[[maybe_unused]] __host__ void __assertfail(
		[[maybe_unused]] char const * __message,
		[[maybe_unused]] char const * __file,
		[[maybe_unused]] unsigned __line,
		[[maybe_unused]] char const * __function,
		[[maybe_unused]] size_t __charSize)
	{
	}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-noreturn"
	// Fix "unresolved extern function".
	[[maybe_unused]] void __assert_fail(
		char const * __message, char const * __file, unsigned __line, char const * __function)
	{
		// See __clang_cuda_runtime_wrapper.h
		__assertfail(__message, __file, __line, __function, sizeof(char));
	}
#pragma clang diagnostic pop
}
#endif
