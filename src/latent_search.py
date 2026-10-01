from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .decoding import build_greedy_decode_function
from .evaluation import build_evaluation_function
from .objective import build_objective_function


class SearchState(NamedTuple):
    population_z: jnp.ndarray       
    population_objective: jnp.ndarray     
    population_layout_keys: jnp.ndarray   
    best_objective: jnp.ndarray           
    best_geometry: jnp.ndarray            
    best_net_grid_span_x: jnp.ndarray    
    best_net_grid_span_y: jnp.ndarray
    best_alignment: jnp.ndarray
    best_overlap: jnp.ndarray


def cosine_anneal(initial_value, final_value, round_index, total_rounds):
    progress = min(1.0, round_index / max(1, total_rounds))
    return final_value + (initial_value - final_value) * 0.5 * (1.0 + np.cos(np.pi * progress))


def build_layout_key(decoded):
    return jnp.concatenate([decoded.block_grid_x, decoded.block_grid_y, decoded.block_grid_w, decoded.block_grid_h], axis=-1)


def select_survivors_by_objective(objective_values, layout_keys, n_survivors, deduplicate):
    ranked = jnp.argsort(-objective_values)
    if not deduplicate:
        return ranked[:n_survivors]

    ranked_keys = layout_keys[ranked]
    is_same_layout = (ranked_keys[:, None, :] == ranked_keys[None, :, :]).all(axis=-1)
    is_earlier = jnp.tril(jnp.ones_like(is_same_layout), k=-1).astype(bool)
    duplicates_a_better_row = (is_same_layout & is_earlier).any(axis=1)

    return ranked[jnp.argsort(duplicates_a_better_row)][:n_survivors]


def build_latent_search_function(configuration, problem_instance):
    decode_latent_batch = build_greedy_decode_function(configuration, problem_instance)
    evaluate_floorplan_batch = build_evaluation_function(problem_instance)
    compute_objective = build_objective_function(configuration, problem_instance)

    latent_dim = int(configuration["ls_dim"])
    population_size = int(configuration["search_population"])
    num_elite = max(2, int(population_size * float(configuration["search_elite_fraction"])))
    num_restarts = int(population_size * float(configuration["search_restart_fraction"]))
    num_children = population_size - num_restarts
    interpolation_low = float(configuration["search_interpolation_low"])
    interpolation_high = float(configuration["search_interpolation_high"])
    deduplicate = bool(configuration["DEDUPLICATE_POPULATION_BY_LAYOUT"])
    protect_restarts = bool(configuration["PROTECT_RESTARTS_FROM_TRUNCATION"])

    @jax.jit
    def decode_and_score(parameters, latent_batch):
        decoded = decode_latent_batch(parameters, latent_batch)
        metrics = evaluate_floorplan_batch(decoded.block_grid_x, decoded.block_grid_y, decoded.block_grid_w, decoded.block_grid_h)
        
        return decoded, metrics, compute_objective(metrics)

    def generate_children(population_z, population_objective, mutation_scale, key):
        parent_a_key, parent_b_key, alpha_key, mutation_key = jax.random.split(key, 4)
        elite_indices = jnp.argsort(-population_objective)[:num_elite]

        parent_a = elite_indices[jax.random.randint(parent_a_key,(num_children,), 0, num_elite)]
        parent_b = elite_indices[jax.random.randint(parent_b_key, (num_children,), 0, num_elite)]

        alpha = (jax.random.uniform(alpha_key, (num_children, 1)) * (interpolation_high - interpolation_low) + interpolation_low)
        children = (alpha * population_z[parent_a] + (1.0 - alpha) * population_z[parent_b])

        return children + mutation_scale * jax.random.normal(mutation_key, (num_children, latent_dim))

    def save_best_geometry(state, candidate_latents, decoded, metrics, candidate_objective):
        best_candidate = jnp.argmax(candidate_objective)
        improved = candidate_objective[best_candidate] > state.best_objective
        pick = lambda new_value, old_value: jnp.where(improved, new_value, old_value)

        candidate_geometry = jnp.stack([decoded.block_grid_x[best_candidate],
                                        decoded.block_grid_y[best_candidate],
                                        decoded.block_grid_w[best_candidate],
                                        decoded.block_grid_h[best_candidate]])
        return state._replace(
            best_objective=pick(candidate_objective[best_candidate], state.best_objective),
            best_geometry=pick(candidate_geometry, state.best_geometry),
            best_net_grid_span_x=pick(metrics["net_grid_span_x"][best_candidate], state.best_net_grid_span_x),
            best_net_grid_span_y=pick(metrics["net_grid_span_y"][best_candidate], state.best_net_grid_span_y),
            best_alignment=pick(metrics["alignment"][best_candidate], state.best_alignment),
            best_overlap=pick(metrics["overlap"][best_candidate], state.best_overlap))

    @jax.jit
    def run_one_round(parameters, state, mutation_scale, key):
        restart_key, generation_key = jax.random.split(key)
        children = generate_children(state.population_z,  state.population_objective, mutation_scale, generation_key)
        fresh_latents = jax.random.normal(restart_key, (num_restarts, latent_dim))
        candidate_latents = jnp.concatenate([children,fresh_latents])

        decoded, metrics, candidate_objective = decode_and_score(parameters, candidate_latents)
        candidate_keys = build_layout_key(decoded)

        pooled_latents = jnp.concatenate([state.population_z, candidate_latents])
        pooled_objective = jnp.concatenate([state.population_objective, candidate_objective])
        pooled_keys = jnp.concatenate([state.population_layout_keys, candidate_keys])

        if protect_restarts:
            restart_slots = jnp.arange( pooled_objective.shape[0] - num_restarts, pooled_objective.shape[0])
            competing = jnp.arange(pooled_objective.shape[0] - num_restarts)
            survivors = competing[select_survivors_by_objective(pooled_objective[competing], pooled_keys[competing], population_size - num_restarts, deduplicate)]
            
            survivors = jnp.concatenate([survivors, restart_slots])
        else:
            survivors = select_survivors_by_objective(pooled_objective, pooled_keys, population_size, deduplicate)

        state = state._replace(population_z=pooled_latents[survivors], population_objective=pooled_objective[survivors],  population_layout_keys=pooled_keys[survivors])

        return save_best_geometry(state, candidate_latents,decoded, metrics, candidate_objective)

    def initialize_search_state(parameters, key):
        population_z = jax.random.normal(key, (population_size, latent_dim))
        decoded, metrics, population_objective = decode_and_score(parameters, population_z)
        empty_state = SearchState(
            population_z = population_z,
            population_objective = population_objective,
            population_layout_keys = build_layout_key(decoded),
            best_objective = jnp.float32(-jnp.inf),
            best_geometry = jnp.zeros((4, problem_instance.num_movable_blocks), dtype=jnp.int32),
            best_net_grid_span_x = jnp.zeros(problem_instance.num_nets, dtype=jnp.int32),
            best_net_grid_span_y = jnp.zeros(problem_instance.num_nets, dtype=jnp.int32),
            best_alignment = jnp.float32(0.0), best_overlap=jnp.float32(0.0))
        
        return save_best_geometry(empty_state, population_z, decoded, metrics, population_objective)

    return run_one_round, initialize_search_state, decode_and_score


def get_count_distinct_layouts(population_layout_keys):
    return len(np.unique(np.asarray(population_layout_keys), axis=0))


def run_latent_search(configuration, problem_instance, parameters, search_seed=None,
                      prebuilt_search_function=None):
    run_one_round, initialize_search_state, decode = (prebuilt_search_function or build_latent_search_function(configuration, problem_instance))

    total_rounds = int(configuration["search_rounds"])
    population_size = int(configuration["search_population"])
    log_every = int(configuration["search_log_every"])
    mutation_initial = float(configuration["search_mutation_initial"])
    mutation_final = float(configuration["search_mutation_final"])

    if search_seed is None:
        search_seed = configuration["search_seed"]
    key = jax.random.PRNGKey(int(search_seed))
    key, initialization_key = jax.random.split(key)

    started_at = time.time()
    state = initialize_search_state(parameters, initialization_key)
    jax.block_until_ready(state.best_objective)

    history = [{"round": 0, "seconds": time.time() - started_at, "evaluations": population_size,
                "best_objective": float(state.best_objective),
                "mean_objective": float(state.population_objective.mean()),
                "distinct_layouts": get_count_distinct_layouts(state.population_layout_keys)}]
    
    print(f"[round 0] J={history[0]['best_objective']:.4f} "
          f"mean={history[0]['mean_objective']:.3f} "
          f"distinct={history[0]['distinct_layouts']:3d} "
          f"{history[0]['seconds']:.0f}s", flush=True)

    for round_index in range(1, total_rounds + 1):
        mutation_scale = cosine_anneal(mutation_initial, mutation_final, round_index,total_rounds)
        key, round_key = jax.random.split(key)
        state = run_one_round(parameters, state, jnp.float32(mutation_scale), round_key)

        if round_index % log_every == 0 or round_index == total_rounds:
            record = {"round": round_index,
                      "seconds": time.time() - started_at,
                      "evaluations": population_size * (round_index + 1),
                      "best_objective": float(state.best_objective),
                      "mean_objective": float(state.population_objective.mean()),
                      "mutation_scale": mutation_scale,
                      "population_latent_std": float(state.population_z.std(axis=0).mean()),
                      "distinct_layouts": get_count_distinct_layouts(state.population_layout_keys)}
            history.append(record)

            print(f"[round {round_index:4d}] J={record['best_objective']:.4f} "
                  f"mean={record['mean_objective']:.3f} "
                  f"sigma={mutation_scale:.3f} "
                  f"popstd={record['population_latent_std']:.3f} "
                  f"distinct={record['distinct_layouts']:3d} "
                  f"{record['seconds']:.0f}s", flush=True)

    return state, history
