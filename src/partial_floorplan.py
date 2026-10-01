from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp

from .evaluation import NET_BOUND_SENTINEL, round_half_to_even_of_half_integer

class PartialFloorplanState(NamedTuple):
    occupancy_corner_deltas: jnp.ndarray   
    block_grid_x: jnp.ndarray              
    block_grid_y: jnp.ndarray              
    block_grid_w: jnp.ndarray              
    block_grid_h: jnp.ndarray              
    net_twice_min_x: jnp.ndarray           
    net_twice_max_x: jnp.ndarray           
    net_twice_min_y: jnp.ndarray
    net_twice_max_y: jnp.ndarray
    net_connector_count: jnp.ndarray       


class FootprintMasks(NamedTuple):
    is_feasible: jnp.ndarray       
    is_bordering: jnp.ndarray      
    n_occupied_cells_covered: jnp.ndarray  
    is_within_bounds: jnp.ndarray      


def initialize_partial_floorplan(problem_instance):
    num_layer = problem_instance.num_layer
    num_grid = problem_instance.num_grid_x
    num_movable_blocks = problem_instance.num_movable_blocks

    has_fixed = problem_instance.net_has_fixed_connector
    fixed_box = problem_instance.net_fixed_bounding_box

    def initial_bound(column, sentinel_sign):
        return jnp.where(has_fixed, 2 * fixed_box[:, column], sentinel_sign * NET_BOUND_SENTINEL).astype(jnp.int32)

    return PartialFloorplanState(
        occupancy_corner_deltas=jnp.zeros((num_layer, num_grid + 1, num_grid + 1), dtype=jnp.float32),
        block_grid_x=jnp.zeros(num_movable_blocks, dtype=jnp.int32),
        block_grid_y=jnp.zeros(num_movable_blocks, dtype=jnp.int32),
        block_grid_w=problem_instance.block_initial_w.astype(jnp.int32),
        block_grid_h=problem_instance.block_initial_h.astype(jnp.int32),
        net_twice_min_x=initial_bound(0, +1),
        net_twice_max_x=initial_bound(1, -1),
        net_twice_min_y=initial_bound(2, +1),
        net_twice_max_y=initial_bound(3, -1),
        net_connector_count=has_fixed.astype(jnp.int32),
    )


def compute_occupancy_and_integral_image(state, num_grid):
    occupancy = jnp.cumsum(jnp.cumsum(state.occupancy_corner_deltas, axis=-1), axis=-2)
    occupancy = occupancy[:, :num_grid, :num_grid]
    integral_image = jnp.cumsum(jnp.cumsum(
        jnp.pad(occupancy, ((0, 0), (1, 0), (1, 0))), axis=-1), axis=-2)
    return occupancy, integral_image


def sum_over_windows(integral_image, index_x_low, index_x_high, index_y_low, index_y_high):
    rows_high = integral_image[index_x_high]                
    rows_low = integral_image[index_x_low]
    return (rows_high[:, index_y_high] - rows_low[:, index_y_high]- rows_high[:, index_y_low] + rows_low[:, index_y_low])


def compute_footprint_masks(integral_image_for_die, footprint_w, footprint_h, num_grid):
    grid_coordinate = jnp.arange(num_grid, dtype=jnp.int32)

    index_x_high = jnp.minimum(grid_coordinate + footprint_w, num_grid)
    index_y_high = jnp.minimum(grid_coordinate + footprint_h, num_grid)
    n_occupied_cells_covered = sum_over_windows(integral_image_for_die, grid_coordinate, index_x_high, grid_coordinate, index_y_high)

    fits_in_x = (grid_coordinate + footprint_w) <= num_grid
    fits_in_y = (grid_coordinate + footprint_h) <= num_grid
    is_within_bounds = fits_in_x[:, None] & fits_in_y[None, :]
    is_feasible = (n_occupied_cells_covered == 0) & is_within_bounds

    expanded_x_low = jnp.maximum(grid_coordinate - 1, 0)
    expanded_y_low = jnp.maximum(grid_coordinate - 1, 0)
    expanded_x_high = jnp.minimum(grid_coordinate + footprint_w + 1, num_grid)
    expanded_y_high = jnp.minimum(grid_coordinate + footprint_h + 1, num_grid)
    ring_sum = sum_over_windows(integral_image_for_die, expanded_x_low, expanded_x_high, expanded_y_low, expanded_y_high) - n_occupied_cells_covered

    touches_die_edge = ((grid_coordinate == 0)[:, None] | (grid_coordinate == 0)[None, :] | ((grid_coordinate + footprint_w) == num_grid)[:, None]
                        | ((grid_coordinate + footprint_h) == num_grid)[None, :])
    
    is_bordering = ((ring_sum > 0) | touches_die_edge) & is_feasible

    return FootprintMasks(is_feasible=is_feasible, is_bordering=is_bordering, n_occupied_cells_covered=n_occupied_cells_covered, is_within_bounds=is_within_bounds)


def has_any_feasible_cell(integral_image_for_die, footprint_w, footprint_h, num_grid):
    grid_coordinate = jnp.arange(num_grid, dtype=jnp.int32)
    index_x_high = jnp.minimum(grid_coordinate + footprint_w, num_grid)
    index_y_high = jnp.minimum(grid_coordinate + footprint_h, num_grid)
    n_occupied_cells_covered = sum_over_windows(integral_image_for_die, grid_coordinate,
                                       index_x_high, grid_coordinate, index_y_high)
    is_within_bounds = ((grid_coordinate + footprint_w) <= num_grid)[:, None] & \
                ((grid_coordinate + footprint_h) <= num_grid)[None, :]
    return ((n_occupied_cells_covered == 0) & is_within_bounds).any()


def compute_wiremask(state, block_index, problem_instance):
    num_grid = problem_instance.num_grid_x
    net_indices = problem_instance.block_net_index[block_index]          
    net_is_real = problem_instance.block_net_valid[block_index]

    net_is_live = net_is_real & (state.net_connector_count[net_indices] > 0)
    net_contribution = (net_is_live.astype(jnp.float32) * problem_instance.net_weight[net_indices])      

    box_min_x = round_half_to_even_of_half_integer(state.net_twice_min_x[net_indices])
    box_max_x = round_half_to_even_of_half_integer(state.net_twice_max_x[net_indices])
    box_min_y = round_half_to_even_of_half_integer(state.net_twice_min_y[net_indices])
    box_max_y = round_half_to_even_of_half_integer(state.net_twice_max_y[net_indices])

    grid_coordinate = jnp.arange(num_grid, dtype=jnp.float32)[None, :]   

    def distance_outside_box(box_low, box_high):
        return (jnp.maximum(box_low[:, None].astype(jnp.float32) - grid_coordinate, 0.0) + jnp.maximum(grid_coordinate - box_high[:, None].astype(jnp.float32), 0.0))

    ramp_x = (distance_outside_box(box_min_x, box_max_x) * net_contribution[:, None]).sum(axis=0)                 
    ramp_y = (distance_outside_box(box_min_y, box_max_y) * net_contribution[:, None]).sum(axis=0)
    return ramp_x[:, None] + ramp_y[None, :]


def normalise_wiremask(wiremask):
    return wiremask / jnp.maximum(wiremask.max(), 1.0)


def compute_alignment_map(state, block_index, footprint_w, footprint_h, step_index, problem_instance):

    num_grid = problem_instance.num_grid_x
    partner_index = problem_instance.alignment_partner_index[block_index]
    partner_exists = partner_index >= 0
    safe_partner_index = jnp.maximum(partner_index, 0)
    partner_is_placed = (problem_instance.decode_order_position[safe_partner_index] < step_index)

    partner_x = state.block_grid_x[safe_partner_index]
    partner_y = state.block_grid_y[safe_partner_index]
    partner_w = state.block_grid_w[safe_partner_index]
    partner_h = state.block_grid_h[safe_partner_index]

    grid_coordinate = jnp.arange(num_grid, dtype=jnp.int32)
    overlap_x = jnp.maximum(jnp.minimum(grid_coordinate + footprint_w, partner_x + partner_w) - jnp.maximum(grid_coordinate, partner_x), 0)
    overlap_y = jnp.maximum(jnp.minimum(grid_coordinate + footprint_h, partner_y + partner_h) - jnp.maximum(grid_coordinate, partner_y), 0)

    projected_overlap = (overlap_x[:, None] * overlap_y[None, :]).astype(jnp.float32)
    return projected_overlap * (partner_exists & partner_is_placed).astype(jnp.float32)


def place_block(state, block_index, grid_x, grid_y, footprint_w, footprint_h, problem_instance):
    die_index = problem_instance.block_die_index[block_index]
    far_x, far_y = grid_x + footprint_w, grid_y + footprint_h

    corner_deltas = state.occupancy_corner_deltas
    for corner_x, corner_y, sign in ((grid_x, grid_y, 1.0), (far_x, grid_y, -1.0), (grid_x, far_y, -1.0), (far_x, far_y, 1.0)):
        corner_deltas = corner_deltas.at[die_index, corner_x, corner_y].add(sign)

    net_indices = problem_instance.block_net_index[block_index]
    net_is_real = problem_instance.block_net_valid[block_index]
    twice_centre_x = 2 * grid_x + footprint_w
    twice_centre_y = 2 * grid_y + footprint_h

    return PartialFloorplanState(
        occupancy_corner_deltas=corner_deltas,
        block_grid_x=state.block_grid_x.at[block_index].set(grid_x),
        block_grid_y=state.block_grid_y.at[block_index].set(grid_y),
        block_grid_w=state.block_grid_w.at[block_index].set(footprint_w),
        block_grid_h=state.block_grid_h.at[block_index].set(footprint_h),

        net_twice_min_x=state.net_twice_min_x.at[net_indices].min(jnp.where(net_is_real, twice_centre_x, NET_BOUND_SENTINEL)),
        net_twice_max_x=state.net_twice_max_x.at[net_indices].max(jnp.where(net_is_real, twice_centre_x, -NET_BOUND_SENTINEL)),
        net_twice_min_y=state.net_twice_min_y.at[net_indices].min(jnp.where(net_is_real, twice_centre_y, NET_BOUND_SENTINEL)),
        net_twice_max_y=state.net_twice_max_y.at[net_indices].max(jnp.where(net_is_real, twice_centre_y, -NET_BOUND_SENTINEL)),
        net_connector_count=state.net_connector_count.at[net_indices].add(net_is_real.astype(jnp.int32))
    )


def build_spatial_maps(state, occupancy, integral_image, step_index, footprint_masks, footprint_w, footprint_h, problem_instance):
    num_grid = problem_instance.num_grid_x
    num_layer = problem_instance.num_layer
    num_movable_blocks = problem_instance.num_movable_blocks

    block_index = problem_instance.decode_order[step_index]

    next_step_index = jnp.minimum(step_index + 1, num_movable_blocks - 1)
    next_step_exists = (step_index + 1) < num_movable_blocks
    next_block_index = problem_instance.decode_order[next_step_index]
    next_die_index = problem_instance.block_die_index[next_block_index]
    next_footprint_masks = compute_footprint_masks(integral_image[next_die_index], problem_instance.block_initial_w[next_block_index], problem_instance.block_initial_h[next_block_index], num_grid)

    keep_next = next_step_exists.astype(jnp.float32)

    channels = [occupancy[die] for die in range(num_layer)]
    channels.append(normalise_wiremask(compute_wiremask(state, block_index,problem_instance)))
    channels.append(footprint_masks.is_feasible.astype(jnp.float32))
    channels.append((footprint_masks.is_feasible & footprint_masks.is_bordering).astype(jnp.float32))
    channels.append(keep_next * normalise_wiremask(compute_wiremask(state, next_block_index, problem_instance)))
    channels.append(keep_next * (next_footprint_masks.is_feasible & next_footprint_masks.is_bordering).astype(jnp.float32))
    channels.append(compute_alignment_map(state, block_index, footprint_w, footprint_h, step_index, problem_instance) / problem_instance.mean_block_area)
    
    return jnp.stack(channels, axis=-1)
