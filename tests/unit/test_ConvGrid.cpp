#include <CL/sycl.hpp>
#ifdef SYCL_DEVICE_ONLY
#undef SYCL_DEVICE_ONLY
#endif
#include <Eigen/Eigen>

#include <catch2/catch.hpp>
#include <convfelt/ConvGrid.hpp>
#include <convfelt/iter.hpp>

SCENARIO("Populaing ConvGrid filter partitions using common data storage")
{
	GIVEN("a ConvGrid of size (10,10,3) with 2x2 filter partitions")
	{
		convfelt::ConvGrid grid{{10, 10, 3}, Felt::Vec2i{2, 2}};

		WHEN("data is queried")
		{
			THEN("it is a column-major FxP matrix")
			{
				auto const & data = grid.matrix();
				STATIC_REQUIRE(!std::decay_t<decltype(data)>::IsRowMajor);
				CHECK(data.rows() == grid.child_size().prod());
				CHECK(data.cols() == grid.children().size().prod());
			}
		}

		WHEN("data is modified within a filter partition")
		{
			grid.children().get({0, 2, 0}).set(1, 123);

			THEN("expected column of parent grid's data is updated")
			{
				CHECK(grid.matrix()(1, 2) == 123);
			}
		}
	}
}