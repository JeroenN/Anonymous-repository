from __future__ import annotations

import jax

from src.checkpoint import load_parameters, read_training_step_from_filename
from src.configuration import (configure_jax_compilation_cache,
                               derive_configuration_fields_from_instances,
                               get_config_path, load_configuration)
from src.problem_instance import load_benchmark_instances
from src.solution_dataset import load_teacher_solutions, split_train_and_validation
from src.training import run_training
from src.vae_model import count_parameters, initialize_vae_parameters


def initialize_or_restore_parameters(configuration, reference_instance):
    checkpoint_path = configuration.get("INITIALIZE_FROM_CHECKPOINT")
    if checkpoint_path:
        return (load_parameters(checkpoint_path, configuration, reference_instance),
                read_training_step_from_filename(checkpoint_path))

    parameters = initialize_vae_parameters(
        jax.random.PRNGKey(int(configuration["initialization_seed"])), configuration,
        reference_instance)
    return parameters, 0


def load_teacher_sets_per_circuit(configuration, problem_instances):
    training_sets, validation_sets = {}, {}
    for circuit_name, instance in problem_instances.items():
        solutions = load_teacher_solutions(
            instance, configuration["TEACHER_DATASET_PATHS"][circuit_name])
        training_sets[circuit_name], validation_sets[circuit_name] = split_train_and_validation(
            solutions, float(configuration["validation_fraction"]),
            int(configuration["validation_split_seed"]))
    return training_sets, validation_sets


def main():
    configuration = load_configuration(get_config_path())
    configure_jax_compilation_cache(configuration)

    problem_instances = load_benchmark_instances(configuration["BENCHMARK_DIRECTORY"], configuration["CIRCUIT_NAMES"])
    configuration = derive_configuration_fields_from_instances(configuration, problem_instances)

    training_sets, validation_sets = load_teacher_sets_per_circuit(configuration, problem_instances)

    reference_instance = next(iter(problem_instances.values()))
    parameters, start_step = initialize_or_restore_parameters(configuration, reference_instance)
    print(f"{count_parameters(parameters):,} parameters, init seed "
          f"{configuration['initialization_seed']}, training seed "
          f"{configuration['training_seed']}, starting at step {start_step}")

    run_training(configuration, problem_instances, parameters, training_sets, validation_sets,
                 start_step=start_step)


if __name__ == "__main__":
    main()
