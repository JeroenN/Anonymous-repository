from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from .transformer import gelu

# Activations NHWC [batch, height, width, channels]
# kernels HWIO [kernel height, kernel width, in_channels, out_channels].
CONVOLUTION_DIMENSION_NUMBERS = ("NHWC", "HWIO", "NHWC")


def resolve_compute_dtype(configuration):
    dtype_name = configuration.get("cell_head_compute_dtype", "float16")
    assert dtype_name in ("float16", "bfloat16", "float32"), \
        f"must be float16, bfloat16 or float32, not {dtype_name!r}"
    return jnp.dtype(dtype_name)


def initialize_convolution_kernel(in_channels, out_channels, kernel_size, key):
    fan_in = in_channels * kernel_size * kernel_size
    limit = jnp.sqrt(1.0 / fan_in)
    kernel = random.uniform(key, (kernel_size, kernel_size, in_channels, out_channels), minval=-limit,maxval=limit)
    return kernel, jnp.zeros(out_channels)


def initialize_transposed_convolution_kernel(in_channels, out_channels, kernel_size, key):
    fan_in = out_channels * kernel_size * kernel_size
    limit = jnp.sqrt(1.0 / fan_in)
    kernel = random.uniform(key, (kernel_size, kernel_size, in_channels, out_channels), minval=-limit, maxval=limit)
    return kernel, jnp.zeros(out_channels)


def convolution_2d(x, convolution_params, compute_dtype):
    kernel, bias = convolution_params
    same_padding = (kernel.shape[0] - 1) // 2

    convolved = jax.lax.conv_general_dilated(x[None].astype(compute_dtype), kernel.astype(compute_dtype),
                                             window_strides=(1, 1),
                                             padding=((same_padding, same_padding), (same_padding, same_padding)),
                                             dimension_numbers=CONVOLUTION_DIMENSION_NUMBERS)[0]
    
    return convolved + bias.astype(compute_dtype)


def transposed_convolution_2d(x, convolution_params, stride, padding, compute_dtype):
    kernel, bias = convolution_params
    equivalent_padding = kernel.shape[0] - 1 - padding
    convolved = jax.lax.conv_general_dilated(x[None].astype(compute_dtype), kernel.astype(compute_dtype),
                                             window_strides=(1, 1),
                                             padding=((equivalent_padding, equivalent_padding), (equivalent_padding, equivalent_padding)),
                                             lhs_dilation=(stride, stride),
                                             dimension_numbers=CONVOLUTION_DIMENSION_NUMBERS)[0]
    
    return convolved + bias.astype(compute_dtype)


def run_convolution_stack(x, stack_params, compute_dtype):
    num_layers = len(stack_params)
    for layer_index, convolution_params in enumerate(stack_params):
        x = convolution_2d(x, convolution_params, compute_dtype)

        if layer_index < num_layers - 1:
            x = gelu(x)

    return x


def run_transposed_convolution_stack(x, stack_params, stride, padding, compute_dtype):
    n_layers = len(stack_params)
    for layer_index, convolution_params in enumerate(stack_params):
        x = transposed_convolution_2d(x, convolution_params,stride, padding, compute_dtype)

        if layer_index < n_layers - 1:
            x = gelu(x)

    return x
