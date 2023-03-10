#pragma once
#include <concepts>
#include <cstddef>

#include "felt2/typedefs.hpp"
#include "felt2/index.hpp"

#include <cppcoro/generator.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/indices.hpp>

namespace convfelt
{
namespace concepts
{
namespace helpers
{
template <class G>
constexpr felt2::Dim DimFor = std::decay_t<G>::Traits::k_dims;

template <class G>
using VecDiFor = felt2::VecDi<DimFor<G>>;
}  // namespace helpers

namespace detail
{
// clang-format off
template <typename T>
concept Grid = requires(T v)
{
	typename helpers::VecDiFor<T>;
	{v.offset()} -> ranges::convertible_to<helpers::VecDiFor<T>>;
	{v.size()} -> ranges::convertible_to<helpers::VecDiFor<T>>;
};

template <typename T>
concept GridWithData = requires(T v)
{
	requires Grid<T>;
	{v.data().size()} -> std::same_as<felt2::PosIdx>;
};
// clang-format on
}  // namespace detail

// clang-format off
template <typename T>
concept Grid = requires(T) { requires detail::Grid<std::decay_t<T>>; };

template <typename T>
concept GridWithData = requires(T) { requires detail::GridWithData<std::decay_t<T>>; };

template <typename T>
concept Integral = requires { ranges::integral<std::decay<T>>; };
// clang-format on
}  // namespace concepts

namespace iter
{
static constexpr auto idx(concepts::Integral auto ... args)
{
	return ranges::views::indices(std::forward<decltype(args)>(args)...);
};

static constexpr auto pos_idx(concepts::GridWithData auto & grid)
{
	return idx(grid.data().size());
};

static inline auto pos(concepts::Grid auto & grid)
	-> cppcoro::generator<concepts::helpers::VecDiFor<decltype(grid)>>
{
	static constexpr auto D = concepts::helpers::DimFor<decltype(grid)>;

	for (auto pos_idx : idx(felt2::PosIdx(grid.size().prod())))
		co_yield grid.offset() + felt2::index<D>(pos_idx, grid.size());
};

template <class G>
using IdxAndPos = std::tuple<felt2::PosIdx, concepts::helpers::VecDiFor<G>>;

static inline auto idx_and_pos(concepts::Grid auto & grid)
	-> cppcoro::generator<IdxAndPos<decltype(grid)>>
{
	static constexpr auto D = concepts::helpers::DimFor<decltype(grid)>;
	using VecDi = concepts::helpers::VecDiFor<decltype(grid)>;

	for (auto const pos_idx : idx(felt2::PosIdx(grid.size().prod())))
	{
		VecDi const pos = grid.offset() + felt2::index<D>(pos_idx, grid.size());
		co_yield {pos_idx, pos};
	}
};

static constexpr decltype(auto) val(concepts::GridWithData auto & grid)
{
	return grid.data();
};

static constexpr decltype(auto) idx_and_val(concepts::GridWithData auto & grid)
{
	return ranges::views::enumerate(grid.data());
};

}  // namespace iter
}  // namespace convfelt