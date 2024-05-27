// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <ranges>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <convfelt/felt2/index.hpp>
#include <convfelt/felt2/typedefs.hpp>

SCENARIO("Pow2 coord conversions")
{
	GIVEN("a pow2 size")
	{
		auto pow2_and_size = GENERATE(
			std::tuple{felt2::PowTwoDu<3>::from_exponents({0, 0, 0}), felt2::Vec3i{1, 1, 1}},
			std::tuple{felt2::PowTwoDu<3>::from_exponents({1, 1, 1}), felt2::Vec3i{2, 2, 2}},
			std::tuple{felt2::PowTwoDu<3>::from_exponents({3, 2, 1}), felt2::Vec3i{8, 4, 2}},
			std::tuple{felt2::PowTwoDu<3>::from_exponents({1, 2, 3}), felt2::Vec3i{2, 4, 8}},
			std::tuple{felt2::PowTwoDu<3>::from_exponents({1, 3, 2}), felt2::Vec3i{2, 8, 4}},
			std::tuple{felt2::PowTwoDu<3>::from_exponents({2, 3, 1}), felt2::Vec3i{4, 8, 2}});

		auto const & pow2_size = std::get<felt2::PowTwoDu<3>>(pow2_and_size);
		auto const & size = std::get<felt2::Vec3i>(pow2_and_size);

		WHEN("Pow2 size is converted to linear vector")
		{
			felt2::Vec3u const vec = pow2_size.as_size();

			THEN("original size is recovered")
			{
				CHECK(vec == size.cast<felt2::Unsigned>());
			}
		}

		AND_GIVEN("a global offset and position")
		{
			auto offset = GENERATE(
				felt2::Vec3i{0, 0, 0},
				felt2::Vec3i{1, 3, -2},
				felt2::Vec3i{1, -2, 3},
				felt2::Vec3i{-2, 1, 3});

			auto positions = std::views::cartesian_product(
								 std::views::iota(offset(0), offset(0) + size(0)),
								 std::views::iota(offset(1), offset(1) + size(1)),
								 std::views::iota(offset(2), offset(2) + size(2))) |
				std::views::transform([](auto const & coords_)
									  { return std::make_from_tuple<felt2::Vec3i>(coords_); });

			auto pos = Catch::Generators::generate(
				"poses",
				CATCH_INTERNAL_LINEINFO,
				[&] {
					return Catch::Generators::makeGenerators(
						Catch::Generators::from_range(positions));
				});

			WHEN("the position is transformed to an index")
			{
				CAPTURE(size, offset, pos, pow2_size.mask(), pow2_size.exps());

				auto idx = felt2::index(pos, size, offset);
				auto pow2_idx = felt2::index(pos, pow2_size, offset);

				THEN("index calculated from linear size is same as index calculated from pow2 size")
				{
					CHECK(idx == pow2_idx);
				}

				AND_WHEN("the index is transformed back into a position")
				{
					CAPTURE(idx, pow2_idx);
					auto expected_pos = felt2::index(idx, size, offset);
					auto actual_pos = felt2::index(idx, pow2_size, offset);

					THEN("original position is recovered")
					{
						CHECK(actual_pos == expected_pos);
						CHECK(expected_pos == pos);
					}
				}
			}
		}
	}

	GIVEN("a target size and expected pow2 size")
	{
		auto target_size_and_expected_pow2_size = GENERATE(
			std::tuple{felt2::Vec3u{0, 0, 0}, felt2::PowTwoDu<3>::from_exponents({0, 0, 0})},
			std::tuple{felt2::Vec3u{1, 1, 1}, felt2::PowTwoDu<3>::from_exponents({0, 0, 0})},
			std::tuple{felt2::Vec3u{2, 2, 2}, felt2::PowTwoDu<3>::from_exponents({1, 1, 1})},
			std::tuple{felt2::Vec3u{3, 3, 3}, felt2::PowTwoDu<3>::from_exponents({2, 2, 2})},
			std::tuple{felt2::Vec3u{4, 4, 4}, felt2::PowTwoDu<3>::from_exponents({2, 2, 2})},
			std::tuple{felt2::Vec3u{5, 5, 5}, felt2::PowTwoDu<3>::from_exponents({3, 3, 3})},
			std::tuple{felt2::Vec3u{0, 1, 2}, felt2::PowTwoDu<3>::from_exponents({0, 0, 1})},
			std::tuple{felt2::Vec3u{3, 4, 5}, felt2::PowTwoDu<3>::from_exponents({2, 2, 3})});

		auto const target_size = std::get<0>(target_size_and_expected_pow2_size);
		WHEN("Pow2 size is constructed from the target size")
		{
			auto const pow2_size = felt2::PowTwoDu<3>::from_minimum_size(target_size);

			THEN("Pow2 size matches expected")
			{
				auto const expected_pow2_size = std::get<1>(target_size_and_expected_pow2_size);

				CAPTURE(target_size, expected_pow2_size.exps(), pow2_size.exps());

				CHECK(pow2_size == expected_pow2_size);
			}
		}
	}
}

SCENARIO("Equality operator for PowTwoDu")
{
	GIVEN("two identical PowTwoDu objects")
	{
		felt2::PowTwoDu<3> pow2_1 = felt2::PowTwoDu<3>::from_exponents({1, 2, 3});
		felt2::PowTwoDu<3> pow2_2 = felt2::PowTwoDu<3>::from_exponents({1, 2, 3});

		THEN("they are equal")
		{
			CHECK(pow2_1 == pow2_2);
		}
	}

	GIVEN("two different PowTwoDu objects")
	{
		felt2::PowTwoDu<3> pow2_1 = felt2::PowTwoDu<3>::from_exponents({1, 2, 3});
		felt2::PowTwoDu<3> pow2_2 = felt2::PowTwoDu<3>::from_exponents({3, 2, 1});

		THEN("they are not equal")
		{
			CHECK(pow2_1 != pow2_2);
		}
	}
}