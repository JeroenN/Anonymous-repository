#Adapted from flexplanner
from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Each block contains 12 features
NUM_BLOCK_STATIC_FEATURES = 12


@dataclass
class ProblemInstance:
    circuit_name: str
    num_grid_x: int
    num_grid_y: int
    num_layer: int
    grid_width: float
    grid_height: float
    aspect_ratio_range: tuple

    num_movable_blocks: int
    block_names: list
    block_die_index: jnp.ndarray              
    block_nominal_area: jnp.ndarray           
    block_initial_w: jnp.ndarray              
    block_initial_h: jnp.ndarray              
    footprint_options: jnp.ndarray            
    footprint_option_valid: jnp.ndarray      
    footprint_option_ratio: jnp.ndarray       
    initial_footprint_option_index: jnp.ndarray  

    alignment_partner_index: jnp.ndarray      

    num_nets: int
    net_block_index: jnp.ndarray              
    net_block_valid: jnp.ndarray              
    net_fixed_bounding_box: jnp.ndarray       
    net_has_fixed_connector: jnp.ndarray     
    net_weight: jnp.ndarray                   

    decode_order: jnp.ndarray                 
    decode_order_position: jnp.ndarray        

    block_net_index: jnp.ndarray              
    block_net_valid: jnp.ndarray              
    footprint_option_features: jnp.ndarray    
    block_static_features: jnp.ndarray        
    maximum_block_area: float
    mean_block_area: float

    @property
    def num_footprint_options(self):
        return int(self.footprint_options.shape[1])

    @property
    def max_nets_per_block(self):
        return int(self.block_net_index.shape[1])


def load_problem_instance(problem_instance_path):
    exported = np.load(problem_instance_path)

    num_movable_blocks = int(exported["M"])
    num_nets = int(exported["N"])

    block_net_index, block_net_valid = build_padded_block_net_lists(
        exported["net_blk"], exported["net_blk_valid"], num_movable_blocks)

    footprint_options = jnp.asarray(exported["wh_opts"], dtype=jnp.int32)
    block_nominal_area = jnp.asarray(exported["area"], dtype=jnp.int32)

    return ProblemInstance(
        circuit_name=str(exported["name"]),
        num_grid_x=int(exported["num_grid_x"]),
        num_grid_y=int(exported["num_grid_y"]),
        num_layer=int(exported["num_layer"]),
        grid_width=float(exported["grid_width"]),
        grid_height=float(exported["grid_height"]),
        aspect_ratio_range=tuple(float(v) for v in exported["ratio_range"]),

        num_movable_blocks=num_movable_blocks,
        block_names=[str(s) for s in exported["block_names"]],
        block_die_index=jnp.asarray(exported["z"], dtype=jnp.int32),
        block_nominal_area=block_nominal_area,
        block_initial_w=jnp.asarray(exported["init_w"], dtype=jnp.int32),
        block_initial_h=jnp.asarray(exported["init_h"], dtype=jnp.int32),
        footprint_options=footprint_options,
        footprint_option_valid=jnp.asarray(exported["wh_valid"], dtype=bool),
        footprint_option_ratio=jnp.asarray(exported["wh_ratio"], dtype=jnp.float32),
        initial_footprint_option_index=jnp.asarray(exported["init_opt"], dtype=jnp.int32),

        alignment_partner_index=jnp.asarray(exported["partner"], dtype=jnp.int32),

        num_nets=num_nets,
        net_block_index=jnp.asarray(exported["net_blk"], dtype=jnp.int32),
        net_block_valid=jnp.asarray(exported["net_blk_valid"], dtype=bool),
        net_fixed_bounding_box=jnp.asarray(exported["net_fixed_bbox"], dtype=jnp.int32),
        net_has_fixed_connector=jnp.asarray(exported["net_has_fixed"], dtype=bool),
        net_weight=jnp.asarray(exported["net_weight"], dtype=jnp.float32),

        decode_order=jnp.asarray(exported["order"], dtype=jnp.int32),
        decode_order_position=jnp.asarray(exported["order_pos"], dtype=jnp.int32),

        block_net_index=block_net_index,
        block_net_valid=block_net_valid,
        footprint_option_features=build_footprint_option_features(
            footprint_options, block_nominal_area, int(exported["num_grid_x"])),
        block_static_features=build_block_static_features(exported, block_net_valid),
        maximum_block_area=float(np.asarray(exported["area"]).max()),
        mean_block_area=float(np.asarray(exported["area"]).mean()),
    )


def load_benchmark_instances(benchmark_directory, circuit_names):
    instances = {}
    for circuit_name in circuit_names:
        path = os.path.join(benchmark_directory, f"problem_{circuit_name}.npz")
        instances[circuit_name] = load_problem_instance(path)
    print("[instances] " + ", ".join( f"{name} (M={instance.num_movable_blocks}, R={instance.num_footprint_options}, "
        f"nets={instance.num_nets})" for name, instance in instances.items()))
    
    return instances


def build_padded_block_net_lists(net_block_index, net_block_valid, num_movable_blocks):
    nets_of_block = [[] for _ in range(num_movable_blocks)]
    for net in range(net_block_index.shape[0]):
        for slot in range(net_block_index.shape[1]):
            if net_block_valid[net, slot]:
                nets_of_block[int(net_block_index[net, slot])].append(net)
    nets_of_block = [sorted(set(nets)) for nets in nets_of_block]

    max_nets_per_block = max(1, max(len(nets) for nets in nets_of_block))
    padded_index = np.zeros((num_movable_blocks, max_nets_per_block), dtype=np.int32)
    padded_valid = np.zeros((num_movable_blocks, max_nets_per_block), dtype=bool)
    for block, nets in enumerate(nets_of_block):
        padded_index[block, :len(nets)] = nets
        padded_valid[block, :len(nets)] = True
    return jnp.asarray(padded_index), jnp.asarray(padded_valid)


def build_footprint_option_features(footprint_options, block_nominal_area, num_grid_x):
    option_w = footprint_options[..., 0].astype(jnp.float32)
    option_h = footprint_options[..., 1].astype(jnp.float32)
    log_aspect_ratio = jnp.log(option_w / option_h)
    area_fraction = (option_w * option_h) / block_nominal_area.reshape(-1, 1).astype(jnp.float32)
    return jnp.stack([option_w / num_grid_x, option_h / num_grid_x, log_aspect_ratio, area_fraction], axis=-1)


def build_block_static_features(exported, block_net_valid):
    num_grid_x, num_grid_y = int(exported["num_grid_x"]), int(exported["num_grid_y"])
    num_layer = int(exported["num_layer"])
    num_movable_blocks = int(exported["M"])
    cells_per_die = float(num_grid_x * num_grid_y)

    area = np.asarray(exported["area"], dtype=np.float64)
    die_index = np.asarray(exported["z"])
    partner_index = np.asarray(exported["partner"])
    has_partner = (partner_index >= 0).astype(np.float64)
    partner_area = np.where(partner_index >= 0, area[np.maximum(partner_index, 0)], 0.0)

    netlist_degree = np.asarray(block_net_valid).sum(axis=1).astype(np.float64)
    total_area_per_die = np.array([area[die_index == die].sum() for die in range(num_layer)])

    # Sorted blocks by area
    area_rank_within_die = np.zeros(num_movable_blocks, dtype=np.float64)

    for die in range(num_layer):
        on_this_die = np.flatnonzero(die_index == die)
        by_descending_area = on_this_die[np.argsort(-area[on_this_die], kind="stable")]
        area_rank_within_die[by_descending_area] = (np.arange(len(on_this_die))/ max(1, len(on_this_die)))

    return jnp.asarray(np.stack([
        area / cells_per_die,
        np.asarray(exported["init_w"], dtype=np.float64) / num_grid_x,
        np.asarray(exported["init_h"], dtype=np.float64) / num_grid_y,
        die_index.astype(np.float64),
        has_partner,
        partner_area / cells_per_die,
        netlist_degree / 32.0,
        np.asarray(exported["order_pos"], dtype=np.float64) / num_movable_blocks,
        np.full(num_movable_blocks, num_movable_blocks / 300.0),
        total_area_per_die[die_index] / cells_per_die,
        np.clip(np.log(area / cells_per_die), -8.0, None) / 8.0,
        area_rank_within_die,
    ], axis=-1), dtype=jnp.float32)


def build_decode_schedule_constants(problem_instance):
    order = problem_instance.decode_order
    num_layer = problem_instance.num_layer
    num_movable_blocks = problem_instance.num_movable_blocks

    die_of_token = problem_instance.block_die_index[order]
    area_of_token = (problem_instance.block_nominal_area[order].astype(jnp.float32) / problem_instance.maximum_block_area)

    die_one_hot = jnp.stack([(die_of_token == die).astype(jnp.float32) for die in range(num_layer)], axis=-1)      

    count_before_token = jnp.cumsum(die_one_hot, axis=0) - die_one_hot      
    blocks_per_die = jnp.stack([(problem_instance.block_die_index == die).sum().astype(jnp.float32) for die in range(num_layer)])             

    return {
        "die_of_token": die_of_token,
        "normalised_area_of_token": area_of_token,
        "normalised_step_index": (jnp.arange(num_movable_blocks, dtype=jnp.float32) / num_movable_blocks),
        "fraction_of_die_placed_before_token": (count_before_token / jnp.maximum(blocks_per_die, 1.0).reshape(1, -1)),
    }
