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
namespace detail
{

template <class G>
using VecDiFor = Felt::VecDi<Felt::Impl::Traits<std::decay_t<G>>::t_dims>;

// clang-format off
template <typename T>
concept Grid = requires(T v)
{
	typename VecDiFor<T>;
	{v.data().size()} -> std::same_as<std::size_t>;
	{v.offset()} -> std::convertible_to<VecDiFor<T>>;
	{v.size()} -> std::convertible_to<VecDiFor<T>>;
};
// clang-format on
}  // namespace detail

// clang-format off
template <typename T>
concept Grid = requires(T) { requires detail::Grid<std::decay_t<T>>; };

template <typename T>
concept Integral = requires { std::integral<std::decay<T>>; };
// clang-format on
}  // namespace concepts

namespace iter
{
static constexpr auto idx(concepts::Integral auto &&... args)
{
	return ranges::views::indices(std::forward<decltype(args)>(args)...);
};

static constexpr auto idx(concepts::Grid auto && grid)
{
	return idx(grid.data().size());
};

static inline auto pos(concepts::Grid auto && grid)
	-> cppcoro::generator<concepts::detail::VecDiFor<decltype(grid)>>
{
	for (auto idx : idx(grid.size().prod()))
		co_yield grid.offset() + Felt::index<3>(Felt::PosIdx(idx), grid.size());
};

static constexpr auto idx_val(concepts::Grid auto && grid)
{
	return ranges::views::enumerate(grid.data());
};

}  // namespace iter
}  // namespace convfelt