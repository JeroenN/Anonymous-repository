from __future__ import annotations

import json
import jax
import numpy as np

from pathlib import Path

from src.checkpoint import load_parameters
from src.configuration import (configure_jax_compilation_cache, derive_configuration_fields_from_instances, get_config_path, load_configuration)
from src.evaluation import compute_original_hpwl_in_double_precision
from src.latent_search import build_latent_search_function, run_latent_search
from src.objective import build_objective_function
from src.problem_instance import load_benchmark_instances
from src.vae_model import count_parameters


def load_model_parameters(configuration, reference_instance):
    return load_parameters(configuration["CHECKPOINT_PATH"], configuration, reference_instance)


def describe_best_layout(problem_instance, net_grid_span_x, net_grid_span_y, alignment, overlap, objective):
    hpwl = float(compute_original_hpwl_in_double_precision(np.asarray(net_grid_span_x), np.asarray(net_grid_span_y), problem_instance))

    return {"hpwl": hpwl, "alignment": float(alignment), "overlap": float(overlap), "objective": float(objective)}


# Report over multiple seeds
def resolve_search_seeds(configuration):
    if configuration.get("search_seeds"):
        return [int(seed) for seed in configuration["search_seeds"]]
    
    return [int(configuration["search_seed"])]


def main():
    configuration = load_configuration(get_config_path())
    configure_jax_compilation_cache(configuration)

    problem_instances = load_benchmark_instances(configuration["BENCHMARK_DIRECTORY"], configuration["CIRCUIT_NAMES"])
    configuration = derive_configuration_fields_from_instances(configuration, problem_instances)

    output_directory = Path(configuration["OUTPUT_DIRECTORY"])
    output_directory.mkdir(parents=True, exist_ok=True)

    reference_instance = next(iter(problem_instances.values()))
    parameters = load_model_parameters(configuration, reference_instance)

    search_seeds = resolve_search_seeds(configuration)
    objective_description = build_objective_function(configuration, reference_instance).description

    print(f"population {configuration['search_population']}, "
          f"{configuration['search_rounds']} rounds, seeds {search_seeds}, "
          f"deduplicate={configuration['DEDUPLICATE_POPULATION_BY_LAYOUT']}, "
          f"protect_restarts={configuration['PROTECT_RESTARTS_FROM_TRUNCATION']}")


    results = {"objective": objective_description, "search_seeds": search_seeds, "per_circuit": {}}
    for circuit_name, problem_instance in problem_instances.items():
        print(f"\n{circuit_name}")
        search_function = build_latent_search_function(configuration, problem_instance)
        per_seed = {}
        for search_seed in search_seeds:
            state, history = run_latent_search(configuration, problem_instance, parameters, search_seed=search_seed, prebuilt_search_function=search_function)
            
            best = describe_best_layout(problem_instance, state.best_net_grid_span_x, state.best_net_grid_span_y, state.best_alignment, state.best_overlap, state.best_objective)
            
            per_seed[str(search_seed)] = {"best": best, "history": history, "decodes": history[-1]["evaluations"], "seconds": history[-1]["seconds"]}

            np.savez(output_directory / f"best_layout_{circuit_name}_seed{search_seed}.npz",
                     block_grid_x=np.asarray(state.best_geometry[0]),
                     block_grid_y=np.asarray(state.best_geometry[1]),
                     block_grid_w=np.asarray(state.best_geometry[2]),
                     block_grid_h=np.asarray(state.best_geometry[3]))
            
            print(f"[{circuit_name} seed {search_seed}] J={best['objective']:.4f} "
                  f"hpwl={best['hpwl']:.1f} alignment={best['alignment']:.4f} "
                  f"overlap={best['overlap']:g} in {history[-1]['seconds']:.0f}s", flush=True)

        objectives = [entry["best"]["objective"] for entry in per_seed.values()]
        results["per_circuit"][circuit_name] = {
            "per_seed": per_seed,
            "objective_mean": float(np.mean(objectives)),
            "objective_std": float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0,
            "objective_min": float(np.min(objectives)),
            "objective_max": float(np.max(objectives))}
        
        entry = results["per_circuit"][circuit_name]
        print(f"[{circuit_name}] over {len(objectives)} seeds: mean {entry['objective_mean']:+.4f}"
              f" sd {entry['objective_std']:.4f} range {entry['objective_min']:+.4f}"
              f" to {entry['objective_max']:+.4f}")

        with open(output_directory / "inference_results.json", "w") as results_file:
            json.dump(results, results_file, indent=2)

    results["mean_objective_over_circuits"] = float(np.mean(
        [entry["objective_mean"] for entry in results["per_circuit"].values()]))

    print("\nsummary, mean over search seeds")
    print(f"{'circuit':8s} {'mean J':>8s} {'sd':>7s} {'min':>8s} {'max':>8s}")
    for circuit_name, entry in results["per_circuit"].items():
        print(f"{circuit_name:8s} {entry['objective_mean']:+8.4f} {entry['objective_std']:7.4f} "
              f"{entry['objective_min']:+8.4f} {entry['objective_max']:+8.4f}")
    print(f"{'mean':8s} {results['mean_objective_over_circuits']:+8.4f}")

    with open(output_directory / "inference_results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"\n{output_directory}/inference_results.json")


if __name__ == "__main__":
    main()
