# Adapted from https://alessiodevoto.github.io/ViT-in-pure-JAX/s
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

LAYER_NORM_EPSILON = 1e-5
ATTENTION_MASK_BIAS = -1e9


def initialize_attention(hidden_dim, num_heads, att_head_dim, key):
    query_key, key_key, value_key, output_key = random.split(key, 4)

    fan_in = hidden_dim
    fan_out = att_head_dim * num_heads
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))

    query_w = random.uniform(query_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    query_b = jnp.zeros(fan_out)
    key_w = random.uniform(key_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    key_b = jnp.zeros(fan_out)
    value_w = random.uniform(value_key, (fan_in, fan_out), minval=-limit, maxval=limit)
    value_b = jnp.zeros(fan_out)

    output_w = random.uniform(output_key, (fan_out, hidden_dim), minval=-limit, maxval=limit)
    output_b = jnp.zeros(hidden_dim)

    return query_w, key_w, value_w, query_b, key_b, value_b, output_w, output_b


def initialize_mlp(hidden_dim, mlp_dim, key):
    first_key, second_key = random.split(key)
    limit = jnp.sqrt(6.0 / (hidden_dim + mlp_dim))

    first_w = random.uniform(first_key, (hidden_dim, mlp_dim), minval=-limit, maxval=limit)
    first_b = jnp.zeros(mlp_dim)
    second_w = random.uniform(second_key, (mlp_dim, hidden_dim), minval=-limit, maxval=limit)
    second_b = jnp.zeros(hidden_dim)

    return first_w, first_b, second_w, second_b


def initialize_mlp_with_widths(input_dim, inner_dim, output_dim, key):
    first_key, second_key = random.split(key)
    first_w, first_b = initialize_linear(input_dim, inner_dim, first_key)
    second_w, second_b = initialize_linear(inner_dim, output_dim, second_key)
    return first_w, first_b, second_w, second_b


def initialize_layer_norm(hidden_dim):
    gamma = jnp.ones(hidden_dim)
    beta = jnp.zeros(hidden_dim)
    return gamma, beta


def initialize_transformer_stack(num_layers, hidden_dim, mlp_dim, num_heads, att_head_dim,
                                 key):
    layer_keys = random.split(key, num_layers)
    return [(initialize_mlp(hidden_dim, mlp_dim, layer_keys[layer_index]),
             initialize_attention(hidden_dim, num_heads, att_head_dim, layer_keys[layer_index]),
             initialize_layer_norm(hidden_dim),
             initialize_layer_norm(hidden_dim))
            for layer_index in range(num_layers)]


def initialize_linear(input_dim, output_dim, key):
    weight = random.normal(key, (input_dim, output_dim)) * jnp.sqrt(1.0 / input_dim)
    return weight, jnp.zeros(output_dim)


def gelu(x):
    return 0.5 * x * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))

def layer_norm(x, layrnorm_params):
    gamma, beta = layrnorm_params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPSILON) + beta


def mlp(x, mlp_params):
    first_w, first_b, second_w, second_b = mlp_params
    up_projection = gelu(jnp.matmul(x, first_w) + first_b)
    return jnp.matmul(up_projection, second_w) + second_b


def linear(x, linear_params):
    weight, bias = linear_params
    return jnp.matmul(x, weight) + bias


def split_into_heads(projected, num_heads, att_head_dim):
    num_tokens = projected.shape[0]
    return projected.reshape(num_tokens, num_heads, att_head_dim).swapaxes(0, 1)


def merge_heads(attended):
    num_heads, num_tokens, att_head_dim = attended.shape
    return attended.swapaxes(0, 1).reshape(num_tokens, num_heads * att_head_dim)


def self_attention(x, attn_params, num_heads, att_head_dim, causal=False):
    query_w, key_w, value_w, query_b, key_b, value_b, output_w, output_b = attn_params
    num_tokens, hidden_dim = x.shape

    query = split_into_heads(jnp.matmul(x, query_w) + query_b, num_heads, att_head_dim)
    key = split_into_heads(jnp.matmul(x, key_w) + key_b, num_heads, att_head_dim)
    value = split_into_heads(jnp.matmul(x, value_w) + value_b, num_heads, att_head_dim)

    attention_logits = (jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(att_head_dim))
    if causal:
        token_positions = jnp.arange(num_tokens)
        allowed = token_positions[:, None] >= token_positions[None, :]
        attention_logits = jnp.where(allowed[None], attention_logits, ATTENTION_MASK_BIAS)

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    attended = jnp.matmul(attention_weights, value)
    return jnp.matmul(merge_heads(attended), output_w) + output_b


def self_attention_with_key_value_cache(query_token, attn_params, cached_keys, cached_values, step_index, num_heads, att_head_dim):
    query_w, key_w, value_w, query_b, key_b, value_b, output_w, output_b = attn_params
    max_num_tokens = cached_keys.shape[0]

    query = (jnp.matmul(query_token, query_w) + query_b).reshape(num_heads, att_head_dim)
    this_key = (jnp.matmul(query_token, key_w) + key_b).reshape(num_heads, att_head_dim)
    this_value = (jnp.matmul(query_token, value_w) + value_b).reshape(num_heads, att_head_dim)

    cached_keys = cached_keys.at[step_index].set(this_key)
    cached_values = cached_values.at[step_index].set(this_value)

    attention_logits = (jnp.einsum("hd,thd->ht", query, cached_keys)
                        / jnp.sqrt(att_head_dim))
    key_is_written = jnp.arange(max_num_tokens) <= step_index
    attention_logits = jnp.where(key_is_written[None], attention_logits, ATTENTION_MASK_BIAS)

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    attended = jnp.einsum("ht,thd->hd", attention_weights, cached_values)
    output = jnp.matmul(attended.reshape(num_heads * att_head_dim), output_w) + output_b

    return output, cached_keys, cached_values


def transformer_block(inp, block_params, num_heads, att_head_dim, causal=False):
    mlp_params, attn_params, ln1_params, ln2_params = block_params

    attended = self_attention(layer_norm(inp, ln1_params), attn_params, num_heads,
                              att_head_dim, causal=causal)
    residual = inp + attended
    return residual + mlp(layer_norm(residual, ln2_params), mlp_params)


def transformer_block_with_key_value_cache(query_token, block_params, cached_keys, cached_values, step_index, num_heads, att_head_dim):
    mlp_params, attn_params, ln1_params, ln2_params = block_params

    attended, cached_keys, cached_values = self_attention_with_key_value_cache(layer_norm(query_token, ln1_params), attn_params, cached_keys, cached_values,
                                                                              step_index, num_heads, att_head_dim)
    residual = query_token + attended
    output = residual + mlp(layer_norm(residual, ln2_params), mlp_params)

    return output, cached_keys, cached_values


def run_transformer_stack(x, stack_params, num_heads, att_head_dim, causal=False):
    for block_params in stack_params:
        x = transformer_block(x, block_params, num_heads, att_head_dim, causal=causal)
    return x


def run_transformer_stack_with_key_value_cache(query_token, stack_params, cached_keys, cached_values, step_index, num_heads, att_head_dim):
    updated_keys, updated_values = cached_keys, cached_values

    for layer_index, block_params in enumerate(stack_params):
        query_token, layer_keys, layer_values = transformer_block_with_key_value_cache(
            query_token, block_params, updated_keys[layer_index],
            updated_values[layer_index], step_index, num_heads, att_head_dim)
        updated_keys = updated_keys.at[layer_index].set(layer_keys)
        updated_values = updated_values.at[layer_index].set(layer_values)

    return query_token, updated_keys, updated_values
