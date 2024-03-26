#pragma once
#include <concepts>
#include <cstddef>

#include "felt2/index.hpp"
#include "felt2/typedefs.hpp"

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
constexpr felt2::Dim k_dim_for = std::decay_t<G>::Traits::k_dims;

template <class G>
using VecDiFor = felt2::VecDi<k_dim_for<G>>;
}  // namespace helpers

namespace detail
{
// clang-format off
template <typename T>
concept Grid = requires(T obj_)
{
	typename helpers::VecDiFor<T>;
	{obj_.offset()} -> ranges::convertible_to<helpers::VecDiFor<T>>;
	{obj_.size()} -> ranges::convertible_to<helpers::VecDiFor<T>>;
};

template <typename T>
concept GridWithStorage = requires(T obj_)
{
	requires Grid<T>;
	{obj_.storage().size()} -> std::same_as<felt2::PosIdx>;
};
// clang-format on
}  // namespace detail

// clang-format off
template <typename T>
concept Grid = requires(T) { requires detail::Grid<std::decay_t<T>>; };

template <typename T>
concept GridWithStorage = requires(T) { requires detail::GridWithStorage<std::decay_t<T>>; };

template <typename T>
concept Integral = requires { ranges::integral<std::decay<T>>; };
// clang-format on
}  // namespace concepts

namespace iter
{
static constexpr auto idx(concepts::Integral auto... args_)
{
	return ranges::views::indices(std::forward<decltype(args_)>(args_)...);
};

static constexpr auto pos_idx(concepts::GridWithStorage auto & grid_)
{
	return idx(grid_.storage().size());
};

// NOLINTNEXTLINE(*-avoid-reference-coroutine-parameters
static inline auto pos(concepts::Grid auto & grid_)
	-> cppcoro::generator<concepts::helpers::VecDiFor<decltype(grid_)>>
{
	for (auto pos_idx : idx(felt2::PosIdx(grid_.size().prod())))
		co_yield grid_.offset() + felt2::index(pos_idx, grid_.size());
};

template <class G>
using IdxAndPos = std::tuple<felt2::PosIdx, concepts::helpers::VecDiFor<G>>;

// NOLINTNEXTLINE(*-avoid-reference-coroutine-parameters
static inline auto idx_and_pos(concepts::Grid auto & grid_)
	-> cppcoro::generator<IdxAndPos<decltype(grid_)>>
{
	using VecDi = concepts::helpers::VecDiFor<decltype(grid_)>;

	for (auto const pos_idx : idx(felt2::PosIdx(grid_.size().prod())))
	{
		VecDi const pos = grid_.offset() + felt2::index(pos_idx, grid_.size());
		co_yield{pos_idx, pos};
	}
};

static constexpr decltype(auto) val(concepts::GridWithStorage auto & grid_)
{
	return grid_.storage();
};

static constexpr decltype(auto) idx_and_val(concepts::GridWithStorage auto & grid_)
{
	return ranges::views::enumerate(grid_.storage());
};

}  // namespace iter
}  // namespace convfelt