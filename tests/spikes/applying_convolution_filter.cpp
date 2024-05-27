// Copyright 2024 David Feltell
// SPDX-License-Identifier: MIT
#include <string_view>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/types.hpp>

#include <convfelt/ConvGrid.hpp>
#include <convfelt/FilterSizeHelper.hpp>
#include <convfelt/iter.hpp>

#include "../sycl_util.hpp"

SCENARIO("Using ConvGrid with SYCL to apply convolution filter to an image")
{
	GIVEN("a simple monochrome image file loaded with 1 pixel of zero padding")
	{
		static constexpr std::string_view k_file_path = CONVFELT_TEST_RESOURCE_DIR "/plus.png";
		OIIO::ImageBuf image{std::string{k_file_path}};	 // NOLINT(misc-const-correctness)
		CAPTURE(image.geterror(false));
		REQUIRE(!image.has_error());
		image.read();

		// False positive:
		// NOLINTNEXTLINE(misc-const-correctness)
		convfelt::InputGrid image_grid{
			convfelt::make_host_context(),
			convfelt::InputGrid::PowTwoDu::from_minimum_size(
				{image.spec().height + 2, image.spec().width + 2, image.spec().nchannels}),
			{0, 0, 0},
			0};

		auto image_grid_spec = image.spec();
		image_grid_spec.width = image_grid.size().as_pos()(1);
		image_grid_spec.height = image_grid.size().as_pos()(0);
		image_grid_spec.nchannels = image_grid.size().as_pos()(2);
		image_grid_spec.format = OIIO::TypeDescFromC<felt2::Scalar>::value();

		// NOLINTNEXTLINE(misc-const-correctness)
		OIIO::ImageBuf image_grid_buf{image_grid_spec, image_grid.storage().data()};
		OIIO::ImageBufAlgo::paste(image_grid_buf, 1, 1, 0, 0, image);

		AND_GIVEN("image is split into filter regions")
		{
			sycl::context const ctx;
			sycl::device const dev{sycl::gpu_selector_v};
			//			sycl::device dev{sycl::cpu_selector_v};
			using FilterGrid =
				convfelt::ConvGridTD<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>;

			FilterSizeHelper const filter_input_sizer{
				felt2::PowTwo3u::from_exponents({2, 2, image_grid.size().exps()(2)}),
				felt2::Vec3i{2, 2, 0}};

			// Compute required size of a grid that contains filter input pixels side-by-side, given
			// a source image.
			auto input_size = filter_input_sizer.input_size_from_source_size(image_grid.size());

			// Create a context tieing together objects in this operation. In particular, useful
			// for logging.
			auto convctx = convfelt::make_device_context(dev, ctx);

			static constexpr auto k_max_log_len = 1024U;
			[[maybe_unused]] auto log_storage = felt2::components::device::Log::make_storage(
				dev, ctx, image_grid.storage().size(), k_max_log_len);
			convctx.logger().set_storage(log_storage);

			// Grid of filter inputs expanded from source image, with child grids of input values
			// for each filter side-by-side. Typically the input grid will be much larger than the
			// source image grid, since many pixels are duplicated, depending on the stride of the
			// filter as it moves across the source image.
			auto filter_input_grid = felt2::device::make_unique_sycl<FilterGrid>(
				dev, ctx, convctx, input_size, filter_input_sizer.filter_window());

			// Loop each individual filter input child.
			for (auto const & [filter_pos_idx, filter] :
				 convfelt::iter::idx_and_val(filter_input_grid->children()))
			{
				// Calculate position in original source image that corresponds to the local (0,0)
				// point in the filter.
				felt2::Vec3i const source_pos_start =
					filter_input_sizer.source_start_pos_from_filter_pos(
						filter_input_grid->children().index(filter_pos_idx));

				// Loop each position within the filter.
				for (felt2::PosIdx const local_pos_idx : convfelt::iter::pos_idx(filter))
				{
					// Calculate the position in the original source image that the current filter
					// pixel corresponds to.
					felt2::Vec3i const source_pos =
						source_pos_start + felt2::index(local_pos_idx, filter.size());

					// source_pos may be out of bounds wrt image_grid because the filter_input_grid
					// can have additional padding due to pow2 sizing.
					if (!image_grid.inside(source_pos))
						continue;

					// Set the filter input value from the corresponding value in the source image.
					filter.set(local_pos_idx, image_grid.get(source_pos));
				}
			}

			WHEN("image is split into filter regions on device")
			{
				auto image_grid_device = felt2::device::make_unique_sycl<
					convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
					dev, ctx, convctx, image_grid.size(), image_grid.offset(), 0.0F);

				image_grid_device->storage().assign(
					image_grid.storage().begin(), image_grid.storage().end());

				auto filter_input_grid_device = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convctx,
					input_size,

					filter_input_sizer.filter_window());
				sycl::range const work_items{image_grid_device->storage().size()};
				sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

				queue.submit(
					[&](sycl::handler & cgh_)
					{
						cgh_.parallel_for<class grid_copy>(
							work_items,
							[filter_input_sizer,
							 image_grid_device = image_grid_device.get(),
							 filter_input_grid_device =
								 filter_input_grid_device.get()](sycl::item<1> const & item_)
							{
								// Thread id == idx of source image pixel channel.
								std::size_t const tid = item_.get_linear_id();

								// Configure logging.
								image_grid_device->context().logger().set_stream(&tid);
								filter_input_grid_device->context().logger().set_stream(&tid);

								// Grid of sub-grids, one sub-grid per filter, each containing the
								// input pixel values for that filter.
								auto & filter_inputs = filter_input_grid_device->children();

								// Index of pixel channel in source image.
								felt2::PosIdx const source_pos_idx = tid;
								// Position of pixel channel in source image.
								felt2::Vec3i const source_pos =
									image_grid_device->index(source_pos_idx);

								// Value of channel for selected pixel channel in source image.
								felt2::Scalar const source_value =
									image_grid_device->get(source_pos_idx);

								// Find all filters that use the current source pixel channel as an
								// input, and compute the position in the filter that the source
								// pixel channel corresponds to, then call a callback for each
								// matched filter input.
								filter_input_sizer.input_pos_from_source_pos(
									image_grid_device->size().as_pos(),
									source_pos,
									[&](felt2::Vec3i const & filter_pos_,
										felt2::Vec3i const & global_pos_) {
										filter_inputs.get(filter_pos_)
											.set(global_pos_, source_value);
									});
							});
					});
				queue.wait_and_throw();

				THEN("device grid matches expected grid")
				{
					REQUIRE(!filter_input_grid_device->context().logger().has_logs());
					REQUIRE(!image_grid_device->context().logger().has_logs());

					CHECK(filter_input_grid->size() == filter_input_grid_device->size());
					CHECK(
						filter_input_grid->children().size() ==
						filter_input_grid_device->children().size());
					CHECK(
						filter_input_grid->storage().size() ==
						filter_input_grid_device->storage().size());
					CHECK(
						filter_input_grid->children().storage().size() ==
						filter_input_grid_device->children().storage().size());

					for (felt2::PosIdx const child_pos_idx :
						 convfelt::iter::pos_idx(filter_input_grid->children()))
					{
						auto const & expected_child =
							filter_input_grid->children().get(child_pos_idx);
						auto const & actual_child =
							filter_input_grid_device->children().get(child_pos_idx);

						CHECK(expected_child.size() == actual_child.size());
						CHECK(expected_child.offset() == actual_child.offset());
						CHECK(expected_child.storage().size() == actual_child.storage().size());
						bool const expected_equals_actual_storage =
							std::ranges::equal(expected_child.storage(), actual_child.storage());
						CHECK(expected_equals_actual_storage);
						if (!expected_equals_actual_storage)
						{
							for (auto const idx :
								 std::views::iota(0U, expected_child.storage().size()))
							{
								auto const pos = expected_child.index(idx);
								auto const expected_val = expected_child.get(idx);
								auto const actual_val = actual_child.get(idx);
								CAPTURE(pos, expected_val, actual_val);
								CHECK(actual_val == expected_val);
							}
						}
					}
				}
			}

			AND_GIVEN("an output image and weight matrix")
			{
				FilterSizeHelper const filter_output_sizer{
					felt2::PowTwo3u::from_exponents({1, 1, 2})};

				auto filter_output_grid = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convctx,

					filter_output_sizer.output_size_from_num_filter_regions(
						filter_input_grid->children().size().as_pos()),

					filter_output_sizer.filter_window());

				REQUIRE(
					filter_output_grid->children().size() == filter_input_grid->children().size());

				convfelt::USMMatrix const usm_weights{
					filter_output_sizer.num_filter_elems(),
					filter_input_sizer.num_filter_elems(),
					dev,
					ctx};

				usm_weights.matrix().setRandom();
				//				Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, 1> wrong{
				//					filter_output_grid->child_size().prod(), 1};
				//				wrong.setConstant(1);

				auto const assert_expected_values = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid->children()))
					{
						felt2::Vec3i const filter_pos =
							filter_input_grid->children().index(filter_pos_idx);
						CAPTURE(filter_pos);

						auto const input_matrix =
							filter_input_grid->children().get(filter_pos_idx).matrix();
						auto const output_matrix =
							filter_output_grid->children().get(filter_pos_idx).matrix();
						CHECK(usm_weights.matrix() * input_matrix == output_matrix);
					}
				};

				WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					queue.submit([pgrid = filter_input_grid.get()](sycl::handler & cgh_)
								 { cgh_.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					queue.submit([pgrid = filter_output_grid.get()](sycl::handler & cgh_)
								 { cgh_.prefetch(pgrid->bytes().data(), pgrid->bytes().size()); });
					queue.submit(
						[&usm_weights](sycl::handler & cgh_)
						{ cgh_.prefetch(usm_weights.bytes().data(), usm_weights.bytes().size()); });

					sycl::nd_range<1> const work_range{
						filter_output_grid->storage().size(),
						filter_output_grid->child_size().as_size().prod()};

					queue.submit(
						[&](sycl::handler & cgh_)
						{
							assert(
								usm_weights.matrix().rows() ==
								filter_output_grid->child_size().as_pos().prod());
							assert(
								usm_weights.matrix().cols() ==
								filter_input_grid->child_size().as_pos().prod());

							sycl::accessor<
								felt2::Scalar,
								1,
								sycl::access::mode::read_write,
								sycl::access::target::local> const input_child_data{
								static_cast<std::size_t>(usm_weights.matrix().cols()), cgh_};

							cgh_.parallel_for<class grid_mult>(
								work_range,
								[filter_input_grid = filter_input_grid.get(),
								 filter_output_grid = filter_output_grid.get(),
								 weights = usm_weights.matrix(),
								 input_child_data](sycl::nd_item<1> const & item_)
								{
									std::size_t const group_id = item_.get_group_linear_id();
									std::size_t const local_id = item_.get_local_linear_id();

									auto & input_child =
										filter_input_grid->children().get(group_id);
									auto & output_child =
										filter_output_grid->children().get(group_id);

									if (input_child.storage().size() != input_child_data.size())
									{
										filter_input_grid->context().logger().log(
											local_id,
											"input_child.storage().size() != "
											"input_child_data.size() i.e. ",
											input_child.storage().size(),
											" != ",
											input_child_data.size(),
											"\n");
										return;
									}
									if (local_id >= output_child.storage().size())
									{
										filter_output_grid->context().logger().log(
											local_id,
											"local_id >= output_child.storage().size() i.e. ",
											local_id,
											" >= ",
											output_child.storage().size(),
											"\n");
										return;
									}

									// TODO(DF): Do we need to construct filter input regions in
									//  global memory only to copy out again into local memory?
									//  I.e. will we need the input grid further down the line?
									//  Otherwise could construct input region here - see next
									//  test.
									item_
										.async_work_group_copy(
											input_child_data.get_pointer(),
											sycl::global_ptr<felt2::Scalar>{
												input_child.storage().data()},
											input_child.storage().size())
										.wait();

									using ColVectorMap = Eigen::Map<
										Eigen::Matrix<felt2::Scalar, Eigen::Dynamic, 1>,
										Eigen::ColMajor>;

									ColVectorMap const input_vec{
										input_child_data.get_pointer().get(), weights.cols()};
									ColVectorMap output_vec{// NOLINT(misc-const-correctness)
															output_child.storage().data(),
															weights.rows()};

									auto const row_idx = static_cast<Eigen::Index>(local_id);
									output_vec(row_idx) = weights.row(row_idx).dot(input_vec);
								});
						});
					queue.wait_and_throw();

					THEN("values are as expected")
					{
						CHECK(!filter_input_grid->context().logger().has_logs());
						CHECK(!filter_output_grid->context().logger().has_logs());
						assert_expected_values();
					}
				}  // WHEN("filter is applied to grid using Eigen in a hand-rolled kernel")

				WHEN("filter is applied in kernel that computes filter input on the fly")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					using TemplateParentGrid = convfelt::TemplateParentGridTD<felt2::Scalar, 3, 1U>;
					auto filter_input_template =
						felt2::device::make_unique_sycl<TemplateParentGrid>(
							queue.get_device(),
							queue.get_context(),
							convctx,
							input_size,
							filter_input_sizer.filter_window());

					auto image_grid_device = felt2::device::make_unique_sycl<
						convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
						queue.get_device(),
						queue.get_context(),
						convctx,
						image_grid.size(),
						image_grid.offset(),
						0.0F);

					image_grid_device->storage().assign(
						image_grid.storage().begin(), image_grid.storage().end());

					auto const filters_per_image =
						filter_output_grid->children().size().as_size().prod();
					auto const elems_per_filter = filter_output_grid->child_size().as_size().prod();

					std::size_t const ideal_elems_per_work_group =
						dev.get_info<sycl::info::device::sub_group_sizes>()[0];
					std::size_t const elems_per_image = elems_per_filter * filters_per_image;

					CHECK(elems_per_image == filter_output_grid->size().as_size().prod());
					CHECK(elems_per_image == filter_output_grid->storage().size());

					constexpr auto k_scalar = [](auto val_)
					{ return static_cast<felt2::Scalar>(val_); };

					felt2::Scalar const ideal_work_groups_per_image =
						k_scalar(elems_per_image) / k_scalar(ideal_elems_per_work_group);

					felt2::Scalar const ideal_filters_per_work_group =
						k_scalar(filters_per_image) / ideal_work_groups_per_image;

					auto const filters_per_work_group =
						static_cast<std::size_t>(std::ceil(ideal_filters_per_work_group));

					felt2::Scalar const work_groups_per_image =
						k_scalar(filters_per_image) / k_scalar(filters_per_work_group);

					felt2::Scalar const elems_per_work_group =
						k_scalar(elems_per_image) / work_groups_per_image;

					auto const num_work_groups =
						static_cast<std::size_t>(std::ceil(work_groups_per_image));
					auto const work_group_size = std::min(
						static_cast<std::size_t>(std::ceil(elems_per_work_group)), elems_per_image);

					CAPTURE(
						dev.get_info<sycl::info::device::local_mem_size>(),
						dev.get_info<sycl::info::device::sub_group_sizes>());
					CHECK(
						dev.get_info<sycl::info::device::local_mem_size>() > sizeof(felt2::Scalar) *
							static_cast<std::size_t>(filter_input_sizer.num_filter_elems()));
					CHECK(
						dev.get_info<sycl::info::device::sub_group_sizes>()[0] <= work_group_size);

					// Check invariants
					CHECK(
						usm_weights.matrix().rows() ==
						filter_output_grid->child_size().as_pos().prod());
					CHECK(
						usm_weights.matrix().rows() <= static_cast<Eigen::Index>(work_group_size));
					CHECK(
						usm_weights.matrix().cols() ==
						filter_input_grid->child_size().as_pos().prod());
					CHECK(usm_weights.matrix().cols() == filter_input_sizer.num_filter_elems());
					// TODO(DF): Technically this check could fail and we still have a valid
					//  situation, just complicated slightly by data boundary condition... or could
					//  it?
					CHECK(
						filter_output_grid->storage().size() == num_work_groups * work_group_size);

					sycl::nd_range<1> const work_range{
						num_work_groups * work_group_size, work_group_size};

					queue.submit(
						[&](sycl::handler & cgh_)
						{
							sycl::local_accessor<felt2::Scalar, 2> const input_child_data{
								sycl::range(
									filters_per_work_group,
									static_cast<std::size_t>(usm_weights.matrix().cols())),
								cgh_};

							cgh_.parallel_for<class dynamic_input>(
								work_range,
								[image_grid = image_grid_device.get(),
								 filter_input_template = filter_input_template.get(),
								 filter_output_grid = filter_output_grid.get(),
								 filters_per_work_group,
								 weights = usm_weights.matrix(),
								 input_child_data,
								 filter_input_sizer,
								 input_size = filter_input_sizer.input_size_from_source_size(
									 image_grid.size()),
								 logger = convctx.m_logger_impl](sycl::nd_item<1> const & item_)
								{
									// Get current group of filters.
									std::size_t const group_id = item_.get_group_linear_id();
									// Get current filter pixel channel index.
									std::size_t const local_id = item_.get_local_linear_id();

									// Configure logging.
									std::size_t const global_id = item_.get_global_id();
									logger.set_stream(&global_id);

									// Number of pixel channels processed by each work group.
									felt2::PosIdx const threads_per_work_group =
										item_.get_local_range(0);

									// Number of pixel channels per filter.
									felt2::PosIdx const threads_per_filter =
										threads_per_work_group / filters_per_work_group;

									// Index of this filter within the current group of filters.
									felt2::PosIdx const filter_idx_offset_for_thread =
										local_id / threads_per_filter;

									// Index of this filter globally.
									felt2::PosIdx const filter_idx_for_thread =
										group_id * filters_per_work_group +
										filter_idx_offset_for_thread;

									auto const filter_input_size =
										static_cast<felt2::PosIdx>(weights.cols());

									// Deliberately copy into private memory the "template" child
									// filter grid for the current filter.
									auto input_child = filter_input_template->children().get(
										filter_idx_for_thread);

									// TODO(DF): For large filters (e.g. fully connected layers)
									//  should fall back to global memory?  Or refactor kernel
									//  somehow?  For fully connected layer, input_child is the same
									//  as image_grid, so no need to sample into local memory. So
									//  perhaps safe to assume either that fully connected case or
									//  small-enough filters for local memory?

									// Configure newly created child to point into memory region
									// shared by this group, offset by the index of the current
									// filter, with a size large enough for the filter's input
									// pixels.
									input_child.storage() = std::span(
										&input_child_data[filter_idx_offset_for_thread][0],
										filter_input_size);

									// Obtain a reference to the region (i.e. child grid) of the
									// output image where the filter results should be written.
									auto & output_child =
										filter_output_grid->children().get(filter_idx_for_thread);

									// Assert invariants.
									if (filter_input_size != input_child.storage().size())
									{
										(void)logger.log(
											"num_cols != input_child.storage().size() i.e. ",
											filter_input_size,
											" != ",
											input_child.storage().size(),
											"\n");
										return;
									}
									if (static_cast<felt2::PosIdx>(weights.rows()) >
										threads_per_filter)
									{
										(void)logger.log(
											"weights.rows()	> elems_per_filter i.e. ",
											weights.rows(),
											" > ",
											threads_per_filter,
											"\n");
										return;
									}

									// Initial index of pixel channel for this thread, within this
									// filter.
									felt2::PosIdx const filter_input_elem_idx_for_thread =
										local_id -
										filter_idx_offset_for_thread * threads_per_filter;

									// Copy from source image into group-local filter input buffer.
									// Using SIMD, each thread loops over the pixels in the filter
									// input buffer, jumping over those that are handled by sibling
									// threads, and populates the corresponding element in the
									// filter input buffer from the source image.
									// Seriously clang-tidy!?:
									// NOLINTNEXTLINE(misc-const-correctness)
									for (felt2::PosIdx simd_col = 0; simd_col < filter_input_size;
										 simd_col += threads_per_filter)
									{
										// Filter input pixel channel index (or column of input
										// region row).
										felt2::PosIdx const col =
											simd_col + filter_input_elem_idx_for_thread;
										if (col >= filter_input_size)
											break;

										// Filter input pixel channel position.
										felt2::Vec3i const pos_in_input_grid =
											input_child.index(col);

										// Source image pixel channel position.

										// Position in source grid may be out of bounds, since
										// input_child uses pow2 sizing.
										if (felt2::Vec3i const pos_in_source_grid =
												filter_input_sizer.source_pos_from_input_pos(
													pos_in_input_grid);
											image_grid->inside(pos_in_source_grid))
										{
											// Update filter input buffer with value from source
											// image.
											input_child.set(
												col, image_grid->get(pos_in_source_grid));
										}
										else
										{
											// Out of domain of the source image, set to zero.
											input_child.set(
												col, static_cast<decltype(input_child)::Leaf>(0));
										}
									}

									// Thread's filter input pixel channel index doubles as the
									// thread's output image pixel channel index. Short-circuit if
									// this is out of bounds of the output image region (child grid)
									// size.
									if (filter_input_elem_idx_for_thread >
										output_child.storage().size())
										return;

									// Ensure the filter input buffer has also been filled by
									// sibling threads.
									item_.barrier(sycl::access::fence_space::local_space);

									// Index within region (child grid) of output image to update
									// with results of applying filter to source image.
									auto const row_idx =
										static_cast<Eigen::Index>(filter_input_elem_idx_for_thread);

									// Apply filter to region of source image. Each thread handles
									// one pixel channel of the output image. That is, one
									// row-column multiplation of the overall matrix operation of
									// filter weights applied to image region, to give a single
									// pixel channel output. The region of the source image that the
									// matrix row is applied to is the filter input buffer, fetched
									// into group shared memory, above.
									output_child.matrix()(row_idx) =
										weights.row(row_idx).dot(input_child.matrix());
								});
						});

					auto const print_logs = [&]
					{
						if (!convctx.logger().has_logs())
							return;
						for (auto const stream_id :
							 std::views::iota(0U, convctx.logger().num_streams()))
						{
							if (auto const txt = convctx.logger().text(stream_id); !txt.empty())
								fmt::print("{}: {}", stream_id, txt);
						}
					};

					try
					{
						queue.wait_and_throw();
					}
					catch (std::exception const & exc)
					{
						fmt::print("ASYNC EXCEPTION(S): {}", exc.what());
						print_logs();
						FAIL("SYCL async exception");
					}

					THEN("values are as expected")
					{
						print_logs();
						CHECK(!convctx.logger().has_logs());
						assert_expected_values();
					}
				}  // WHEN("filter is applied in kernel that computes filter input on the fly")

				WHEN("filter is applied to grid using oneMKL")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					oneapi::mkl::blas::column_major::gemm(
						queue,
						oneapi::mkl::transpose::nontrans,
						oneapi::mkl::transpose::nontrans,
						// m: Rows of output / rows of weights
						filter_output_grid->matrix().rows(),
						// n: Columns of output / columns of input
						filter_output_grid->matrix().cols(),
						// k: Rows of input / columns of weights
						filter_input_grid->matrix().rows(),
						// alpha
						1,
						// a: weights
						usm_weights.matrix().data(),
						usm_weights.matrix().rows(),
						// b: input
						filter_input_grid->matrix().data(),
						filter_input_grid->matrix().rows(),
						// beta
						0,
						// c: output
						filter_output_grid->matrix().data(),
						filter_output_grid->matrix().rows());

					queue.wait_and_throw();
					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}
			}  // AND_GIVEN("an output image and weight matrix")

			AND_GIVEN("an output vector and weight matrix")
			{
				FilterSizeHelper const filter_output_sizer{
					felt2::PowTwo3u::from_exponents({0, 0, 3})};

				auto filter_output_grid = felt2::device::make_unique_sycl<FilterGrid>(
					dev,
					ctx,
					convfelt::make_device_context(dev, ctx),
					filter_output_sizer.output_size_from_num_filter_regions(felt2::Vec3i::Ones()),
					filter_output_sizer.filter_window());

				REQUIRE(filter_output_grid->children().size().as_pos() == felt2::Vec3i(1, 1, 1));

				convfelt::USMMatrix const usm_weights{
					filter_output_sizer.num_filter_elems(),
					image_grid.size().as_pos().prod(),
					dev,
					ctx};

				usm_weights.matrix().setRandom();

				auto const assert_expected_values = [&]
				{
					for (auto const & filter_pos_idx :
						 convfelt::iter::pos_idx(filter_output_grid->children()))
					{
						felt2::Vec3i const filter_pos =
							filter_input_grid->children().index(filter_pos_idx);
						CAPTURE(filter_pos);

						auto const input_matrix = image_grid.matrix();
						auto const output_matrix = filter_output_grid->matrix();
						CAPTURE(usm_weights.matrix() * input_matrix, output_matrix);
						auto const diff = usm_weights.matrix() * input_matrix - output_matrix;
						felt2::Scalar const diff_sq = diff.dot(diff);
						CHECK(diff_sq == Catch::Approx(0).margin(1e-6));
					}
				};

				WHEN("filter is applied to grid using oneMKL")
				{
					sycl::queue queue{ctx, dev, &async_handler};  // NOLINT(misc-const-correctness)

					auto image_grid_device = felt2::device::make_unique_sycl<
						convfelt::ByValue<felt2::Scalar, 3, convfelt::GridFlag::is_device_shared>>(
						dev,
						ctx,
						convfelt::make_device_context(dev, ctx),
						image_grid.size(),
						image_grid.offset(),
						0.0F);

					image_grid_device->storage().assign(
						image_grid.storage().begin(), image_grid.storage().end());

					oneapi::mkl::blas::column_major::gemm(
						queue,
						oneapi::mkl::transpose::nontrans,
						oneapi::mkl::transpose::nontrans,
						// m: Rows of output / rows of weights
						filter_output_grid->matrix().rows(),
						// n: Columns of output / columns of input
						filter_output_grid->matrix().cols(),
						// k: Rows of input / columns of weights
						image_grid_device->matrix().rows(),
						// alpha
						1,
						// a: weights
						usm_weights.matrix().data(),
						usm_weights.matrix().rows(),
						// b: input
						image_grid_device->matrix().data(),
						image_grid_device->matrix().rows(),
						// beta
						0,
						// c: output
						filter_output_grid->matrix().data(),
						filter_output_grid->matrix().rows());

					queue.wait_and_throw();
					THEN("values are as expected")
					{
						assert_expected_values();
					}
				}
			}
		}  // AND_GIVEN("image is split into filter regions")
	}
}
