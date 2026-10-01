# Makes sure that during teacher forcing the blocks are placed in the right order and put in the right format 
# This file also handles loss calculation
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .partial_floorplan import (build_spatial_maps, compute_footprint_masks, compute_occupancy_and_integral_image, initialize_partial_floorplan, place_block)
from .problem_instance import build_decode_schedule_constants
from .vae_model import (build_teacher_forced_token_states, compute_cell_logits, compute_footprint_option_logits)


def build_teacher_forced_scoring_function(configuration, problem_instance):
    schedule_constants = build_decode_schedule_constants(problem_instance)
    decode_order = problem_instance.decode_order
    block_die_index = problem_instance.block_die_index
    num_grid = problem_instance.num_grid_x
    num_movable_blocks = problem_instance.num_movable_blocks

    # 0 means decode everything at once
    step_chunk_size = int(configuration.get("teacher_forcing_step_chunk_size", 0) or num_movable_blocks)

    step_chunk_size = min(step_chunk_size, num_movable_blocks)
    num_step_chunks = -(-num_movable_blocks // step_chunk_size) 
    padded_num_steps = num_step_chunks * step_chunk_size

    def replay_teachr_layout(block_grid_x, block_grid_y, block_grid_w, block_grid_h):
        def replay_one_block(floorplan_state, step_index):
            block_index = decode_order[step_index]
            die_index = block_die_index[block_index]
            target_w, target_h = block_grid_w[block_index], block_grid_h[block_index]

            occupancy, integral_image = compute_occupancy_and_integral_image(floorplan_state, num_grid)
            footprint_masks = compute_footprint_masks(integral_image[die_index], target_w, target_h, num_grid)
            spatial_maps = build_spatial_maps(floorplan_state, occupancy, integral_image, step_index, footprint_masks, target_w, target_h, problem_instance)

            floorplan_state = place_block(floorplan_state, block_index, block_grid_x[block_index], block_grid_y[block_index], target_w, target_h, problem_instance)
            return floorplan_state, (spatial_maps, footprint_masks.is_feasible)

        _, replayed = lax.scan(replay_one_block,initialize_partial_floorplan(problem_instance),jnp.arange(num_movable_blocks, dtype=jnp.int32))
        return lax.stop_gradient(replayed)

    def score_one_step(parameters, token_state, step_index, spatial_maps, feasable_mask, target_flat_cell, target_option_index):
        block_index = decode_order[step_index]

        option_logits = compute_footprint_option_logits(parameters, configuration,
                                                        problem_instance, token_state,
                                                        block_index)
        option_loss = -jax.nn.log_softmax(option_logits)[target_option_index]
        option_is_correct = jnp.argmax(option_logits) == target_option_index

        cell_logits = compute_cell_logits(parameters, configuration, problem_instance,
                                          token_state, block_index, target_option_index,
                                          spatial_maps).reshape(-1)

        target_was_illegal = jnp.logical_not(feasable_mask.reshape(-1)[target_flat_cell])
        allowed_cells = feasable_mask.reshape(-1).at[target_flat_cell].set(True)
        cell_logits = jnp.where(allowed_cells, cell_logits, -jnp.inf)

        cell_loss = -jax.nn.log_softmax(cell_logits)[target_flat_cell]
        cell_is_correct = jnp.argmax(cell_logits) == target_flat_cell

        return (cell_loss, option_loss, cell_is_correct.astype(jnp.float32), option_is_correct.astype(jnp.float32), target_was_illegal.astype(jnp.float32))

    def score_teacher_layout(parameters, latent, block_grid_x, block_grid_y, block_grid_w, block_grid_h, footprint_option_index):
        spatial_maps_per_step, feasable_mask_per_step = replay_teachr_layout(block_grid_x, block_grid_y, block_grid_w, block_grid_h)

        token_states = build_teacher_forced_token_states(parameters, configuration, problem_instance, schedule_constants, latent, block_grid_x, block_grid_y, block_grid_w, block_grid_h)

        target_flat_cell_per_step = ((block_grid_x[decode_order] * num_grid + block_grid_y[decode_order]).astype(jnp.int32))
        target_option_per_step = footprint_option_index[decode_order]

        padded_step_index = jnp.minimum(jnp.arange(padded_num_steps, dtype=jnp.int32), num_movable_blocks - 1)
        step_is_real = (jnp.arange(padded_num_steps) < num_movable_blocks).astype(jnp.float32)

        rematerialised_score_one_step = jax.checkpoint(score_one_step)

        def score_one_chunk(chunk):
            return jax.vmap(rematerialised_score_one_step, in_axes=(None, 0, 0, 0, 0, 0, 0))(parameters, *chunk)

        chunked = jax.tree_util.tree_map(
            lambda array: array[padded_step_index].reshape(num_step_chunks, step_chunk_size,*array.shape[1:]),
            (token_states, jnp.arange(num_movable_blocks, dtype=jnp.int32),
             spatial_maps_per_step, feasable_mask_per_step, target_flat_cell_per_step,
             target_option_per_step))

        (cell_loss, option_loss, cell_is_correct, option_is_correct,
         target_was_illegal) = jax.tree_util.tree_map(
            lambda array: array.reshape(-1), jax.lax.map(score_one_chunk, chunked))

        def mean_over_real_steps(per_step):
            return (per_step * step_is_real).sum() / num_movable_blocks

        return {
            "cell_loss": mean_over_real_steps(cell_loss),
            "option_loss": mean_over_real_steps(option_loss),
            "cell_accuracy": mean_over_real_steps(cell_is_correct),
            "option_accuracy": mean_over_real_steps(option_is_correct),
            "illegal_target_fraction": mean_over_real_steps(target_was_illegal),
        }

    return score_teacher_layout
