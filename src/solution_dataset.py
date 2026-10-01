# Dataset loader VAE is trained on
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np



@partial(jax.tree_util.register_dataclass, data_fields=["block_grid_x", "block_grid_y", "block_grid_w", "block_grid_h", "footprint_option_index"], meta_fields=[])
@dataclass
class SolutionSet:
    block_grid_x: jnp.ndarray
    block_grid_y: jnp.ndarray
    block_grid_w: jnp.ndarray
    block_grid_h: jnp.ndarray
    footprint_option_index: jnp.ndarray

    def __len__(self):
        return int(self.block_grid_x.shape[0])

    def take(self, solution_indices):
        return SolutionSet(
            block_grid_x=self.block_grid_x[solution_indices],
            block_grid_y=self.block_grid_y[solution_indices],
            block_grid_w=self.block_grid_w[solution_indices],
            block_grid_h=self.block_grid_h[solution_indices],
            footprint_option_index=self.footprint_option_index[solution_indices])


def load_teacher_solutions(problem_instance, teacher_dataset_paths):
    if isinstance(teacher_dataset_paths, str):
        teacher_dataset_paths = [teacher_dataset_paths]

    assert all(teacher_dataset_paths), (f"TEACHER_DATASET_PATHS for {problem_instance.circuit_name}is empty ")

    geometry_per_file = []
    for path in teacher_dataset_paths:
        assert os.path.exists(path), (f"teacher file for {problem_instance.circuit_name} does not exist: {path}")
        exported = np.load(path)
        stored_block_names = [str(name) for name in exported["names"]]
        assert stored_block_names == list(problem_instance.block_names), f"{path}: block order differs from the {problem_instance.circuit_name} instance"
        geometry_per_file.append(exported["geo"].astype(np.int32))

    geometry = np.concatenate(geometry_per_file, axis=0)        
    block_grid_x, block_grid_y = geometry[:, :, 0], geometry[:, :, 1]
    block_grid_w, block_grid_h = geometry[:, :, 2], geometry[:, :, 3]

    footprint_option_index = resolve_footprint_option_indices(problem_instance, block_grid_w, block_grid_h)

    print(f"{problem_instance.circuit_name}: {geometry.shape[0]} teacher layouts from {len(teacher_dataset_paths)} file(s)")
    return SolutionSet(block_grid_x=jnp.asarray(block_grid_x),
                       block_grid_y=jnp.asarray(block_grid_y),
                       block_grid_w=jnp.asarray(block_grid_w),
                       block_grid_h=jnp.asarray(block_grid_h),
                       footprint_option_index=jnp.asarray(footprint_option_index))


def resolve_footprint_option_indices(problem_instance, block_grid_w, block_grid_h,chunk_size=4096):
    option_w = np.asarray(problem_instance.footprint_options[:, :, 0]) 
    option_h = np.asarray(problem_instance.footprint_options[:, :, 1])
    resolved = np.zeros(block_grid_w.shape, dtype=np.int32)

    for start in range(0, block_grid_w.shape[0], chunk_size):
        chunk_w = block_grid_w[start:start + chunk_size][:, :, None]
        chunk_h = block_grid_h[start:start + chunk_size][:, :, None]
        matches = (option_w[None] == chunk_w) & (option_h[None] == chunk_h)

        assert matches.any(-1).all(), "a teacher footprint is outside the enumerated option set"

        resolved[start:start + chunk_size] = matches.argmax(-1)

    return resolved


def split_train_and_validation(solution_set, validation_fraction, split_seed):
    random_generator = np.random.default_rng(split_seed)
    permutation = random_generator.permutation(len(solution_set))
    num_validation = max(64, int(validation_fraction * len(solution_set)))
    validation_set = solution_set.take(jnp.asarray(permutation[:num_validation]))
    training_set = solution_set.take(jnp.asarray(permutation[num_validation:]))
    print(f"[data] train={len(training_set)} validation={len(validation_set)}")
    return training_set, validation_set
