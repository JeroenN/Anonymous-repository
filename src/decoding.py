from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .partial_floorplan import (build_spatial_maps, compute_footprint_masks, compute_occupancy_and_integral_image, has_any_feasible_cell,
                                initialize_partial_floorplan, place_block)

from .problem_instance import build_decode_schedule_constants
from .transformer import linear

from .vae_model import (build_single_decoder_token_feature, build_single_step_token_state, compute_block_identety, compute_cell_logits,
                        compute_footprint_option_logits, initialize_key_value_cache)

# If there is no place on the die where a block can be placed without overlap, then this penality is subtracted from n_occupied_cells_covered. 
# This way the least bad position with overlap is chosen
OVERLAP_FALLBACK_PENALTY = 1e3


class DecodeCarry(NamedTuple):
    floorplan_state: object
    cached_keys: jnp.ndarray
    cached_values: jnp.ndarray
    previous_placement: jnp.ndarray            
    filled_fraction_per_die: jnp.ndarray       
    num_infeasible_steps: jnp.ndarray          
    num_footprint_retries: jnp.ndarray         


class DecodeOutput(NamedTuple):
    block_grid_x: jnp.ndarray                  
    block_grid_y: jnp.ndarray
    block_grid_w: jnp.ndarray
    block_grid_h: jnp.ndarray
    num_infeasible_steps: jnp.ndarray          
    num_footprint_retries: jnp.ndarray        


def build_greedy_decode_function(configuration, problem_instance):
    schedule_constants = build_decode_schedule_constants(problem_instance)
    block_die_index = problem_instance.block_die_index
    decode_order = problem_instance.decode_order
    footprint_options = problem_instance.footprint_options
    num_layer = problem_instance.num_layer
    num_grid = problem_instance.num_grid_x
    num_movable_blocks = problem_instance.num_movable_blocks
    maximum_block_area = problem_instance.maximum_block_area
    max_footprint_retries = int(configuration["max_footprint_retries"])

    def choose_placeable_footprint(option_logits, block_index, integral_image_for_die):
        def footprint_has_room(option_index):
            return has_any_feasible_cell(
                integral_image_for_die,
                footprint_options[block_index, option_index, 0],
                footprint_options[block_index, option_index, 1], num_grid)

        first_option_index = jnp.argmax(option_logits)

        def retry_is_needed(retry_carry):
            attempt_count, _, _, footprint_fits = retry_carry
            return jnp.logical_and(attempt_count < max_footprint_retries,
                                   jnp.logical_not(footprint_fits))

        def try_next_footprint(retry_carry):
            attempt_count, option_index, option_logits, footprint_fits = retry_carry
            option_logits = option_logits.at[option_index].set(-jnp.inf)
            option_index = jnp.where(footprint_fits, option_index, jnp.argmax(option_logits))
            return (attempt_count + 1, option_index, option_logits, footprint_has_room(option_index))

        attempt_count, option_index, _, _ = lax.while_loop(
            retry_is_needed, try_next_footprint,
            (jnp.int32(0), first_option_index, option_logits,
             footprint_has_room(first_option_index)))
        return option_index, attempt_count

    def decode_one_block(parameters, projected_latent, block_identity_per_step, carry,
                         step_index):
        block_index = decode_order[step_index]
        die_index = block_die_index[block_index]

        token_feature = build_single_decoder_token_feature(schedule_constants, step_index, carry.previous_placement, carry.filled_fraction_per_die)
        token_state, cached_keys, cached_values = build_single_step_token_state(
            parameters, configuration, token_feature, step_index, block_identity_per_step,
            projected_latent, carry.cached_keys, carry.cached_values)

        occupancy, integral_image = compute_occupancy_and_integral_image(carry.floorplan_state, num_grid)
        integral_image_for_die = integral_image[die_index]

        # footprint
        option_logits = compute_footprint_option_logits(parameters, configuration,problem_instance, token_state, block_index)
        option_index, attempt_count = choose_placeable_footprint(option_logits, block_index, integral_image_for_die)
        footprint_w = footprint_options[block_index, option_index, 0]
        footprint_h = footprint_options[block_index, option_index, 1]

        # cell
        footprint_masks = compute_footprint_masks(integral_image_for_die, footprint_w, footprint_h, num_grid)
        spatial_maps = build_spatial_maps(carry.floorplan_state, occupancy, integral_image, step_index, footprint_masks, 
                                          footprint_w, footprint_h, problem_instance)
        cell_logits = compute_cell_logits(parameters, configuration, problem_instance,
                                          token_state, block_index, option_index, spatial_maps)

        # If no placement is possible, the block  is placed on an illegal cell position. This is reported then
        no_feasible_cell = jnp.logical_not(footprint_masks.is_feasible.any())
        cell_logits = jnp.where(no_feasible_cell, cell_logits - OVERLAP_FALLBACK_PENALTY * footprint_masks.n_occupied_cells_covered,cell_logits)

        allowed_cells = jnp.where(no_feasible_cell, footprint_masks.is_within_bounds,footprint_masks.is_feasible)
        cell_logits = jnp.where(allowed_cells, cell_logits, -jnp.inf)

        flat_cell_index = jnp.argmax(cell_logits.reshape(-1))
        grid_x = (flat_cell_index // num_grid).astype(jnp.int32)
        grid_y = (flat_cell_index % num_grid).astype(jnp.int32)

        # state
        floorplan_state = place_block(carry.floorplan_state, block_index, grid_x, grid_y, footprint_w, footprint_h, problem_instance)
        placed_area = (footprint_w * footprint_h).astype(jnp.float32)

        previous_placement = jnp.stack([
            grid_x.astype(jnp.float32) / num_grid, grid_y.astype(jnp.float32) / num_grid,
            footprint_w.astype(jnp.float32) / num_grid,
            footprint_h.astype(jnp.float32) / num_grid,
            die_index.astype(jnp.float32), placed_area / maximum_block_area])
        
        filled_fraction_per_die = carry.filled_fraction_per_die.at[die_index].add(placed_area / float(num_grid * num_grid))

        return DecodeCarry(
            floorplan_state=floorplan_state,
            cached_keys=cached_keys, cached_values=cached_values,
            previous_placement=previous_placement,
            filled_fraction_per_die=filled_fraction_per_die,
            num_infeasible_steps=carry.num_infeasible_steps + no_feasible_cell,
            num_footprint_retries=carry.num_footprint_retries + attempt_count), None

    def decode_single_latent(parameters, z):
        cached_keys, cached_values = initialize_key_value_cache(configuration,
                                                               problem_instance)
        initial_carry = DecodeCarry(
            floorplan_state=initialize_partial_floorplan(problem_instance),
            cached_keys=cached_keys, cached_values=cached_values,
            previous_placement=jnp.zeros(6, dtype=jnp.float32),
            filled_fraction_per_die=jnp.zeros(num_layer, dtype=jnp.float32),
            num_infeasible_steps=jnp.int32(0),
            num_footprint_retries=jnp.int32(0))

        projected_z = linear(z, parameters["latent_input_projection"])
        block_identity_per_step = compute_block_identety(parameters, problem_instance)

        final_carry, _ = lax.scan(
            lambda carry, step_index: decode_one_block(parameters, projected_z,
                                                      block_identity_per_step, carry,
                                                      step_index),
            initial_carry, jnp.arange(num_movable_blocks, dtype=jnp.int32))

        return DecodeOutput(
            block_grid_x=final_carry.floorplan_state.block_grid_x,
            block_grid_y=final_carry.floorplan_state.block_grid_y,
            block_grid_w=final_carry.floorplan_state.block_grid_w,
            block_grid_h=final_carry.floorplan_state.block_grid_h,
            num_infeasible_steps=final_carry.num_infeasible_steps,
            num_footprint_retries=final_carry.num_footprint_retries)

    @jax.jit
    def decode_latent_batch(parameters, latent_batch):
        return jax.vmap(decode_single_latent, in_axes=(None, 0))(parameters, latent_batch)

    return decode_latent_batch
