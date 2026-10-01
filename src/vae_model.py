from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from .convolution import (initialize_convolution_kernel,
                          initialize_transposed_convolution_kernel, resolve_compute_dtype,
                          run_convolution_stack, run_transposed_convolution_stack)
from .problem_instance import NUM_BLOCK_STATIC_FEATURES
from .transformer import (initialize_layer_norm, initialize_linear,
                          initialize_mlp_with_widths, initialize_transformer_stack,
                          layer_norm, linear, mlp, run_transformer_stack,
                          run_transformer_stack_with_key_value_cache)

# Adapted from flexplanner, starts with 16-channel 8x8 map, then 8 -> 16 -> 32 -> 64 -> 128.
DECONVOLUTION_START_CHANNELS = 16
DECONVOLUTION_START_RESOLUTION = 8

DECONVOLUTION_CHANNEL_PLAN = [16, 16, 8, 8]
DECONVOLUTION_KERNEL_SIZE = 4
DECONVOLUTION_STRIDE = 2
DECONVOLUTION_PADDING = 1

NUM_ENCODER_INPUT_FEATURES = 8


def initialize_vae_parameters(key, configuration, problem_instance):
    hidden_dim = configuration["hidden_dim_encoder"]

    parameters = {}

    block_identity_key, positional_key, key = random.split(key, 3)
    parameters["block_identity_mlp"] = initialize_mlp_with_widths(NUM_BLOCK_STATIC_FEATURES, configuration["block_identity_hidden_dim"], hidden_dim, block_identity_key)
    
    parameters["positional_encoding"] = random.normal(positional_key, (configuration["max_num_movable_blocks"], hidden_dim)) * 0.02

    
    encoder_input_key, encoder_stack_key, key = random.split(key, 3)
    parameters["encoder_input_projection"] = initialize_linear(NUM_ENCODER_INPUT_FEATURES, hidden_dim, encoder_input_key)

    parameters["encoder_blocks"] = initialize_transformer_stack(configuration["num_layers_encoder"], hidden_dim, configuration["mlp_dim_encoder"], configuration["num_heads_encoder"], 
                                                                configuration["att_head_dim_encoder"], encoder_stack_key)
    
    parameters["encoder_final_layer_norm"] = initialize_layer_norm(hidden_dim)

    latent_mean_key, latent_log_variance_key, key = random.split(key, 3)
    parameters["latent_mean_projection"] = initialize_linear( hidden_dim, configuration["ls_dim"], latent_mean_key)
    parameters["latent_log_variance_projection"] = initialize_linear(hidden_dim, configuration["ls_dim"], latent_log_variance_key)

    latent_input_key, decoder_input_key, decoder_stack_key, key = random.split(key, 4)
    parameters["latent_input_projection"] = initialize_linear(configuration["ls_dim"], hidden_dim, latent_input_key)
    parameters["decoder_input_projection"] = initialize_linear(configuration["num_token_features"], hidden_dim, decoder_input_key)
    parameters["decoder_blocks"] = initialize_transformer_stack(configuration["num_layers_decoder"], hidden_dim, configuration["mlp_dim_decoder"], configuration["num_heads_decoder"], 
                                                                configuration["att_head_dim_decoder"], decoder_stack_key)
    parameters["decoder_final_layer_norm"] = initialize_layer_norm(hidden_dim)


    footprint_mlp_key, footprint_query_key, footprint_bias_key, key = random.split(key, 4)
    parameters["footprint_option_mlp"] = initialize_mlp_with_widths(4, configuration["footprint_option_embedding_dim"], hidden_dim, footprint_mlp_key)

    parameters["footprint_query_projection"] = initialize_linear(hidden_dim, hidden_dim, footprint_query_key)

    parameters["footprint_option_bias_head"] = initialize_linear(hidden_dim, 1, footprint_bias_key)

    deconvolution_projection_key, key = random.split(key)
    parameters["cell_deconvolution_projection"] = initialize_linear(hidden_dim, DECONVOLUTION_START_CHANNELS * DECONVOLUTION_START_RESOLUTION ** 2, deconvolution_projection_key)

    deconvolution_channels = (DECONVOLUTION_CHANNEL_PLAN + [configuration["num_deconvolution_channels"]])

    parameters["cell_deconvolution_kernels"] = []
    for in_channels, out_channels in zip(deconvolution_channels[:-1], deconvolution_channels[1:]):
        layer_key, key = random.split(key)
        parameters["cell_deconvolution_kernels"].append(initialize_transposed_convolution_kernel(in_channels, out_channels, DECONVOLUTION_KERNEL_SIZE, layer_key))

    local_channels = [configuration["num_spatial_map_channels"],
                      configuration["num_local_convolution_channels"],
                      configuration["num_local_convolution_channels"],
                      configuration["num_deconvolution_channels"]]
    
    parameters["cell_local_convolution_kernels"] = []
    for in_channels, out_channels in zip(local_channels[:-1], local_channels[1:]):
        layer_key, key = random.split(key)
        parameters["cell_local_convolution_kernels"].append(
            initialize_convolution_kernel(in_channels, out_channels, 3, layer_key))

    fusion_channels = [2 * configuration["num_deconvolution_channels"], 8, 1]
    fusion_kernel_sizes = [3, 1]
    parameters["cell_fusion_convolution_kernels"] = []
    for in_channels, out_channels, kernel_size in zip(fusion_channels[:-1],fusion_channels[1:], fusion_kernel_sizes):
        layer_key, key = random.split(key)
        parameters["cell_fusion_convolution_kernels"].append(initialize_convolution_kernel(in_channels, out_channels, kernel_size, layer_key))

    return parameters


def count_parameters(parameters):
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(parameters))


def compute_block_identety(parameters, problem_instance):
    return mlp(problem_instance.block_static_features[problem_instance.decode_order], parameters["block_identity_mlp"])


def build_encoder_input_features(problem_instance, block_grid_x, block_grid_y, block_grid_w, block_grid_h):
    order = problem_instance.decode_order
    num_grid = problem_instance.num_grid_x

    ordered_x = block_grid_x[order].astype(jnp.float32)
    ordered_y = block_grid_y[order].astype(jnp.float32)
    ordered_w = block_grid_w[order].astype(jnp.float32)
    ordered_h = block_grid_h[order].astype(jnp.float32)
    nominal_area = problem_instance.block_nominal_area[order].astype(jnp.float32)

    return jnp.stack([
        ordered_x / num_grid, ordered_y / num_grid,
        ordered_w / num_grid, ordered_h / num_grid,
        (ordered_w * ordered_h) / nominal_area,
        problem_instance.block_die_index[order].astype(jnp.float32),
        (problem_instance.alignment_partner_index[order] >= 0).astype(jnp.float32),
        nominal_area / problem_instance.maximum_block_area], axis=-1)


def encode_floorplan(parameters, configuration, problem_instance, block_grid_x, block_grid_y, block_grid_w, block_grid_h):

    tokens = (linear(build_encoder_input_features(problem_instance, block_grid_x, block_grid_y, block_grid_w, block_grid_h), parameters["encoder_input_projection"])
              + compute_block_identety(parameters, problem_instance) + parameters["positional_encoding"][:problem_instance.num_movable_blocks])

    tokens = run_transformer_stack(tokens, parameters["encoder_blocks"], configuration["num_heads_encoder"], configuration["att_head_dim_encoder"], causal=False)

    pooled = layer_norm(tokens, parameters["encoder_final_layer_norm"]).mean(axis=0)

    latent_mean = linear(pooled, parameters["latent_mean_projection"])
    latent_log_variance = jnp.clip(linear(pooled, parameters["latent_log_variance_projection"]), -8.0, 8.0)

    return latent_mean, latent_log_variance


def reparameterize(latent_mean, latent_log_variance, key):
    return latent_mean + random.normal(key, latent_mean.shape) * jnp.exp(0.5 * latent_log_variance)


def compute_kl_divergence_with_free_bits(latent_mean, latent_log_variance, free_bits):
    per_dimension = 0.5 * (latent_mean ** 2 + jnp.exp(latent_log_variance) - 1.0 - latent_log_variance)
    return jnp.maximum(per_dimension, free_bits).sum(), per_dimension.sum()


def build_decoder_token_features(problem_instance, schedule_constants, block_grid_x, block_grid_y, block_grid_w, block_grid_h):
    order = problem_instance.decode_order
    num_grid = problem_instance.num_grid_x
    num_layer = problem_instance.num_layer
    num_movable_blocks = problem_instance.num_movable_blocks

    ordered_x = block_grid_x[order].astype(jnp.float32)
    ordered_y = block_grid_y[order].astype(jnp.float32)
    ordered_w = block_grid_w[order].astype(jnp.float32)
    ordered_h = block_grid_h[order].astype(jnp.float32)
    ordered_area = ordered_w * ordered_h
    die_of_token = schedule_constants["die_of_token"]

    placement_of_token = jnp.stack([
        ordered_x / num_grid, ordered_y / num_grid,
        ordered_w / num_grid, ordered_h / num_grid,
        die_of_token.astype(jnp.float32),
        ordered_area / problem_instance.maximum_block_area,
    ], axis=-1)                                                       
    previous_placement = jnp.concatenate([jnp.zeros((1, 6), dtype=jnp.float32),placement_of_token[:-1]], axis=0)


    die_one_hot = jnp.stack([(die_of_token == die).astype(jnp.float32) for die in range(num_layer)], axis=-1)    
    area_on_die = ordered_area[:, None] * die_one_hot                

    filled_fraction_per_die = ((jnp.cumsum(area_on_die, axis=0) - area_on_die) / float(num_grid * num_grid))

    current_step = jnp.stack([schedule_constants["normalised_step_index"], die_of_token.astype(jnp.float32), schedule_constants["normalised_area_of_token"]], axis=-1)

    return jnp.concatenate([previous_placement, current_step, schedule_constants["fraction_of_die_placed_before_token"], filled_fraction_per_die], axis=-1)


def build_single_decoder_token_feature(schedule_constants, step_index, previous_placement, filled_fracion_per_die):
    current_step = jnp.stack([schedule_constants["normalised_step_index"][step_index],
                              schedule_constants["die_of_token"][step_index].astype(jnp.float32),
                              schedule_constants["normalised_area_of_token"][step_index]])
    
    return jnp.concatenate([
        previous_placement,
        current_step,
        schedule_constants["fraction_of_die_placed_before_token"][step_index],
        filled_fracion_per_die,
    ])


def build_teacher_forced_token_states(parameters, configuration, problem_instance, schedule_constants, latent, block_grid_x, block_grid_y, block_grid_w, block_grid_h):
    tokens = (compute_block_identety(parameters, problem_instance) + parameters["positional_encoding"][:problem_instance.num_movable_blocks] + 
              linear(build_decoder_token_features(problem_instance, schedule_constants, block_grid_x, block_grid_y, block_grid_w, block_grid_h), parameters["decoder_input_projection"])
              + linear(latent, parameters["latent_input_projection"])[None, :])

    tokens = run_transformer_stack(tokens, parameters["decoder_blocks"], configuration["num_heads_decoder"], configuration["att_head_dim_decoder"], causal=True)

    return layer_norm(tokens, parameters["decoder_final_layer_norm"])


def build_single_step_token_state(parameters, configuration, token_feature, step_index, block_identity_per_step, projected_latent, cached_keys, cached_values):
    token = (block_identity_per_step[step_index] + parameters["positional_encoding"][step_index] + linear(token_feature, parameters["decoder_input_projection"]) + projected_latent)

    token, cached_keys, cached_values = run_transformer_stack_with_key_value_cache( token, parameters["decoder_blocks"], cached_keys, cached_values, step_index,
                                                                                   configuration["num_heads_decoder"], configuration["att_head_dim_decoder"])
    
    return layer_norm(token, parameters["decoder_final_layer_norm"]), cached_keys, cached_values


def initialize_key_value_cache(configuration, problem_instance):
    cache_shape = (configuration["num_layers_decoder"], problem_instance.num_movable_blocks, configuration["num_heads_decoder"], configuration["att_head_dim_decoder"])
    
    return jnp.zeros(cache_shape, dtype=jnp.float32), jnp.zeros(cache_shape, dtype=jnp.float32)


def compute_footprint_option_logits(parameters, configuration, problem_instance, token_state, block_index):
    option_embeddings = mlp(problem_instance.footprint_option_features[block_index], parameters["footprint_option_mlp"])     
    logits = (jnp.matmul(linear(token_state, parameters["footprint_query_projection"]), option_embeddings.T) / jnp.sqrt(configuration["hidden_dim_decoder"]))
    logits = logits + linear(option_embeddings, parameters["footprint_option_bias_head"]).squeeze(-1)

    return jnp.where(problem_instance.footprint_option_valid[block_index], logits, -jnp.inf)


def compute_cell_logits(parameters, configuration, problem_instance, token_state, block_index, footprint_option_index, spatial_maps):
    compute_dtype = resolve_compute_dtype(configuration)

    option_features = problem_instance.footprint_option_features[block_index, footprint_option_index]
    conditioned_state = token_state + mlp(option_features, parameters["footprint_option_mlp"])

    coarse = linear(conditioned_state, parameters["cell_deconvolution_projection"]).reshape(DECONVOLUTION_START_CHANNELS, DECONVOLUTION_START_RESOLUTION, 
                                                                                            DECONVOLUTION_START_RESOLUTION).transpose(1, 2, 0)
    
    latent_preference_map = run_transposed_convolution_stack(coarse, parameters["cell_deconvolution_kernels"], DECONVOLUTION_STRIDE,
                                                             DECONVOLUTION_PADDING, compute_dtype)

    canvas_map = run_convolution_stack(spatial_maps, parameters["cell_local_convolution_kernels"], compute_dtype)

    fused = run_convolution_stack(jnp.concatenate([latent_preference_map, canvas_map], axis=-1),  parameters["cell_fusion_convolution_kernels"], compute_dtype)
    
    return fused[:, :, 0].astype(jnp.float32)
