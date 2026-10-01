# Evaluates on HPWL, Alignment, and overlap. Same evaluation as in FlexPlanner
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

NET_BOUND_SENTINEL = 1 << 20


def round_half_to_even_of_half_integer(twice_the_value):
    floor_half = jnp.floor_divide(twice_the_value, 2)
    is_exact_half = (twice_the_value-2 * floor_half) == 1
    floor_half_is_odd = jnp.mod(floor_half, 2) != 0
    return floor_half + jnp.where(is_exact_half & floor_half_is_odd, 1, 0)


def build_evaluation_function(problem_instance):
    net_block_index = problem_instance.net_block_index
    net_block_valid = problem_instance.net_block_valid
    net_fixed_bounding_box = problem_instance.net_fixed_bounding_box
    net_has_fixed_connector = problem_instance.net_has_fixed_connector
    net_is_live = net_block_valid.any(axis=1) | net_has_fixed_connector

    block_has_partner = problem_instance.alignment_partner_index >= 0
    partner_index = jnp.maximum(problem_instance.alignment_partner_index, 0)
    number_of_partnered_blocks = float(np.asarray(block_has_partner).sum())

    block_die_index = problem_instance.block_die_index
    num_grid = problem_instance.num_grid_x
    num_layer = problem_instance.num_layer
    num_movable_blocks = problem_instance.num_movable_blocks

    def compute_net_grid_spans(block_x, block_y, block_w, block_h):
        twice_centre_x = 2 * block_x + block_w                       
        twice_centre_y = 2 * block_y + block_h

        centre_x_per_slot = twice_centre_x[net_block_index]            
        centre_y_per_slot = twice_centre_y[net_block_index]

        minimum_twice_x = jnp.where(net_block_valid, centre_x_per_slot,NET_BOUND_SENTINEL).min(axis=-1)
        maximum_twice_x = jnp.where(net_block_valid, centre_x_per_slot, -NET_BOUND_SENTINEL).max(axis=-1)
        minimum_twice_y = jnp.where(net_block_valid, centre_y_per_slot, NET_BOUND_SENTINEL).min(axis=-1)
        maximum_twice_y = jnp.where(net_block_valid, centre_y_per_slot, -NET_BOUND_SENTINEL).max(axis=-1)

        minimum_twice_x = jnp.where(net_has_fixed_connector, jnp.minimum(minimum_twice_x, 2 * net_fixed_bounding_box[:, 0]), minimum_twice_x)
        maximum_twice_x = jnp.where(net_has_fixed_connector, jnp.maximum(maximum_twice_x, 2 * net_fixed_bounding_box[:, 1]), maximum_twice_x)
        minimum_twice_y = jnp.where(net_has_fixed_connector, jnp.minimum(minimum_twice_y, 2 * net_fixed_bounding_box[:, 2]), minimum_twice_y)
        maximum_twice_y = jnp.where(net_has_fixed_connector, jnp.maximum(maximum_twice_y, 2 * net_fixed_bounding_box[:, 3]), maximum_twice_y)

        span_x = (round_half_to_even_of_half_integer(maximum_twice_x) - round_half_to_even_of_half_integer(minimum_twice_x))
        span_y = (round_half_to_even_of_half_integer(maximum_twice_y) - round_half_to_even_of_half_integer(minimum_twice_y))
        span_x = jnp.where(net_is_live, jnp.maximum(span_x, 0), 0)
        span_y = jnp.where(net_is_live, jnp.maximum(span_y, 0), 0)
        return span_x.astype(jnp.int32), span_y.astype(jnp.int32)

    def compute_alignment_score(block_x, block_y, block_w, block_h):
        partner_x, partner_y = block_x[partner_index], block_y[partner_index]
        partner_w, partner_h = block_w[partner_index], block_h[partner_index]

        overlap_x = jnp.maximum(jnp.minimum(block_x + block_w, partner_x + partner_w) - jnp.maximum(block_x, partner_x), 0)
        overlap_y = jnp.maximum(jnp.minimum(block_y + block_h, partner_y + partner_h) - jnp.maximum(block_y, partner_y), 0)

        intersection_area = (overlap_x * overlap_y).astype(jnp.float32)
        required_area = jnp.maximum(jnp.minimum(partner_w* partner_h, block_w * block_h), 1).astype(jnp.float32)

        per_block_score = jnp.clip(intersection_area / required_area, 0.0, 1.0)

        return (per_block_score * block_has_partner).sum() / max(1.0, number_of_partnered_blocks)

    def compute_occupancy_canvas(block_x, block_y, block_w, block_h):
        corner_accumulator = jnp.zeros((num_layer, num_grid + 1, num_grid + 1),dtype=jnp.float32)
        far_x, far_y = block_x + block_w, block_y + block_h
        for corner_x, corner_y, sign in ((block_x, block_y, 1.0), (far_x, block_y, -1.0), (block_x, far_y, -1.0), (far_x, far_y, 1.0)):
            corner_accumulator = corner_accumulator.at[ block_die_index, corner_x, corner_y].add(jnp.full((num_movable_blocks,), sign, dtype=jnp.float32))
        occupancy = jnp.cumsum(jnp.cumsum(corner_accumulator, axis=-1), axis=-2)
        return occupancy[:, :num_grid, :num_grid]

    def compute_overlap_fraction(block_x, block_y, block_w, block_h):
        occupancy = compute_occupancy_canvas(block_x, block_y, block_w, block_h)
        return jnp.maximum(occupancy - 1.0, 0.0).sum() / (num_grid * num_grid)

    physical_grid_width = float(problem_instance.grid_width)
    physical_grid_height = float(problem_instance.grid_height)

    def evaluate_single_floorplan(block_x, block_y, block_w, block_h):
        net_span_x, net_span_y = compute_net_grid_spans(block_x, block_y, block_w, block_h)
        total_span_x, total_span_y = net_span_x.sum(), net_span_y.sum()
        return {
            "net_grid_span_x": net_span_x,
            "net_grid_span_y": net_span_y,
            "total_grid_span_x": total_span_x,
            "total_grid_span_y": total_span_y,
            "grid_hpwl": (total_span_x + total_span_y).astype(jnp.float32),
            "hpwl": (total_span_x.astype(jnp.float32) * physical_grid_width + total_span_y.astype(jnp.float32) * physical_grid_height),
            "alignment": compute_alignment_score(block_x, block_y, block_w, block_h),
            "overlap": compute_overlap_fraction(block_x, block_y, block_w, block_h),
        }

    @jax.jit
    def evaluate_floorplan_batch(block_x, block_y, block_w, block_h):
        return jax.vmap(evaluate_single_floorplan)(block_x.astype(jnp.int32),block_y.astype(jnp.int32), block_w.astype(jnp.int32), block_h.astype(jnp.int32))

    return evaluate_floorplan_batch


def compute_original_hpwl_in_double_precision(net_grid_span_x,net_grid_span_y, problem_instance):
    span_x = np.asarray(net_grid_span_x, dtype=np.float64)
    span_y = np.asarray(net_grid_span_y, dtype=np.float64)
    return (span_x * problem_instance.grid_width+ span_y * problem_instance.grid_height).sum(axis=-1)
