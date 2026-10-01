from __future__ import annotations

import os
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from .vae_model import initialize_vae_parameters


def save_parameters(parameters, checkpoint_path):
    payload = {f"parameter_leaf_{index:04d}": np.asarray(leaf)
               for index, leaf in enumerate(jax.tree_util.tree_leaves(parameters))}
    temporary_path = checkpoint_path + ".partial.npz"
    np.savez(temporary_path, **payload)
    os.replace(temporary_path, checkpoint_path)


SNAPSHOT_FILENAME_PATTERN = re.compile(r"^snapshot_(\d+)$")


# gets training step from checkpoint from filename
def read_training_step_from_filename(checkpoint_path):
    match = SNAPSHOT_FILENAME_PATTERN.match(pathlib.Path(checkpoint_path).stem)
    return int(match.group(1)) if match else 0


def load_parameters(checkpoint_path, configuration, problem_instance):
    template_parameters = initialize_vae_parameters(jax.random.PRNGKey(0), configuration,
                                                    problem_instance)
    template_leaves, tree_definition = jax.tree_util.tree_flatten(template_parameters)

    exported = np.load(checkpoint_path)
    restored_leaves = []
    for index, template_leaf in enumerate(template_leaves):
        restored_leaf = exported[f"parameter_leaf_{index:04d}"]
        assert restored_leaf.shape == template_leaf.shape, (
            f"{checkpoint_path}: leaf {index} has shape {restored_leaf.shape}, but this "
            f"config wants {template_leaf.shape}")
        restored_leaves.append(jnp.asarray(restored_leaf))

    return jax.tree_util.tree_unflatten(tree_definition, restored_leaves)
