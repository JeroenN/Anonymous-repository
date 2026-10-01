from __future__ import annotations

import argparse
from pathlib import Path

import jax
import yaml


MINIMUM_COMPILE_SECONDS_WORTH_CACHING = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keys for paths described in the configs
PATH_VALUED_CONFIGURATION_KEYS = (
    "BENCHMARK_DIRECTORY",
    "OUTPUT_DIRECTORY",
    "CHECKPOINT_PATH",
    "INITIALIZE_FROM_CHECKPOINT",
    "TEACHER_DATASET_PATHS",
    "SOURCE_TEACHER_PATHS",
    "JAX_COMPILATION_CACHE_DIR",
)


def get_config_path():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", default="configs/infer_n100.yaml", help="path to the yaml config describing the whole run")

    return argument_parser.parse_args().config


def configure_jax_compilation_cache(configuration):
    cache_directory = configuration.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_directory:
        return configuration

    Path(cache_directory).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", MINIMUM_COMPILE_SECONDS_WORTH_CACHING)

    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return configuration


def get_absolute_path(path):
    if not path:
        return path
    return str(path) if Path(path).is_absolute() else str(PROJECT_ROOT / path)


def resolve_configuration_paths(configuration):
    for key in PATH_VALUED_CONFIGURATION_KEYS:
        value = configuration.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            configuration[key] = get_absolute_path(value)
        elif isinstance(value, list):
            configuration[key] = [get_absolute_path(entry) for entry in value]
        elif isinstance(value, dict):
            configuration[key] = {name: [get_absolute_path(entry) for entry in paths]
                                  for name, paths in value.items()}
    return configuration


def load_configuration(configuration_filename):
    configuration_file = Path(get_absolute_path(configuration_filename))
    with open(configuration_file, "r") as config:
        configuration = yaml.safe_load(config)
    configuration = resolve_configuration_paths(configuration)
    return validate_and_derive_configuration_fields(configuration)

# Assertions that check for valid hyperparameters and calculation of
# att_head_dim_encoder and att_head_dim_decoder.
def validate_and_derive_configuration_fields(configuration):
    if "hidden_dim_encoder" not in configuration:
        return configuration

    assert configuration["hidden_dim_encoder"] % configuration["num_heads_encoder"] == 0, \
        (f"hidden_dim_encoder ({configuration['hidden_dim_encoder']}) must be divisible by "
         f"num_heads_encoder ({configuration['num_heads_encoder']})")
    assert configuration["hidden_dim_decoder"] % configuration["num_heads_decoder"] == 0, \
        (f"hidden_dim_decoder ({configuration['hidden_dim_decoder']}) must be divisible by "
         f"num_heads_decoder ({configuration['num_heads_decoder']})")

    configuration["att_head_dim_encoder"] = (configuration["hidden_dim_encoder"] // configuration["num_heads_encoder"])
    configuration["att_head_dim_decoder"] = (configuration["hidden_dim_decoder"] // configuration["num_heads_decoder"])

    assert configuration["hidden_dim_encoder"] == configuration["hidden_dim_decoder"], \
        "the block identity and positional tables are shared, so both towers need one width"

    return configuration

#num_token_features is 6 (the previous blocks placement) + 3 (which block this step is and how far through the schedule it is) 
# + 2 * num_layer (how full each die is, and how far through each of the dies queue the decode is).
def derive_configuration_fields_from_instances(configuration, problem_instances):
    layer_counts = {name: instance.num_layer for name, instance in problem_instances.items()}
    assert len(set(layer_counts.values())) == 1, f"every instance must have the same die count for one model to span them: {layer_counts}"

    any_instance = next(iter(problem_instances.values()))
    configuration["num_token_features"] = 9 + 2 * any_instance.num_layer

    largest_instance = max(problem_instances.values(),
                           key=lambda instance: instance.num_movable_blocks)
    assert largest_instance.num_movable_blocks <= configuration["max_num_movable_blocks"], (
        f"max_num_movable_blocks={configuration['max_num_movable_blocks']} is too small for "
        f"{largest_instance.circuit_name}, which has {largest_instance.num_movable_blocks} blocks")
    return configuration
