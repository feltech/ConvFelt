#pragma once
#include <concepts>
#include <cstddef>

#include <Felt/Impl/Common.hpp>
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
constexpr Felt::Dim DimFor = Felt::Impl::Traits<std::decay_t<G>>::t_dims;

template <class G>
using VecDiFor = Felt::VecDi<DimFor<G>>;
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
	{v.data().size()} -> std::same_as<Felt::PosIdx>;
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

	for (auto pos_idx : idx(Felt::PosIdx(grid.size().prod())))
		co_yield grid.offset() + Felt::index<D>(pos_idx, grid.size());
};

template <class G>
using IdxAndPos = std::tuple<Felt::PosIdx, concepts::helpers::VecDiFor<G>>;

static inline auto idx_and_pos(concepts::Grid auto & grid)
	-> cppcoro::generator<IdxAndPos<decltype(grid)>>
{
	static constexpr auto D = concepts::helpers::DimFor<decltype(grid)>;
	using VecDi = concepts::helpers::VecDiFor<decltype(grid)>;

	for (auto const pos_idx : idx(Felt::PosIdx(grid.size().prod())))
	{
		VecDi const pos = grid.offset() + Felt::index<D>(pos_idx, grid.size());
		co_yield {pos_idx, pos};
	}
};

static constexpr decltype(auto) val(concepts::GridWithData auto & grid)
{
	return grid.data();
};

static constexpr auto idx_and_val(concepts::GridWithData auto & grid)
{
	return ranges::views::enumerate(grid.data());
};

}  // namespace iter
}  // namespace convfelt