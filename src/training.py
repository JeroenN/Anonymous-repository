from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoint import save_parameters
from .decoding import build_greedy_decode_function
from .evaluation import build_evaluation_function, compute_original_hpwl_in_double_precision
from .objective import build_objective_function
from .teacher_forcing import build_teacher_forced_scoring_function
from .vae_model import (compute_kl_divergence_with_free_bits, encode_floorplan, reparameterize)


def create_learning_rate_schedule(configuration):
    max_lr = float(configuration["learning_rate"])
    warmup_steps = int(configuration["learning_rate_warmup_steps"])
    total_steps = int(configuration["num_training_steps"])

    def learning_rate_at_step(step):
        warmup_factor = jnp.minimum(1.0, (step + 1) / warmup_steps)
        cosine_progress = jnp.minimum(1.0, step / total_steps)
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_progress)) * 0.9 + 0.1
        return max_lr * warmup_factor * cosine_factor

    return learning_rate_at_step


def create_optimizer(configuration):
    adamw = optax.adamw(learning_rate=create_learning_rate_schedule(configuration), b1=0.9, b2=0.95, weight_decay=float(configuration["weight_decay"]))
    return optax.chain(optax.clip_by_global_norm(float(configuration["gradient_clip_norm"])), adamw)


def build_training_step_function(configuration, problem_instance, optimizer):
    score_teacher_layout = build_teacher_forced_scoring_function(configuration,
                                                                problem_instance)
    free_bits = float(configuration["free_bits"])
    option_loss_weight = float(configuration["option_loss_weight"])
    kl_weight = float(configuration["kl_weight"])

    def compute_loss(params, batch, key):
        encode_batch = jax.vmap(lambda x, y, w, h: encode_floorplan(params, configuration, problem_instance,x, y, w, h))

        latent_mean, latent_log_variance = encode_batch(batch.block_grid_x, batch.block_grid_y, batch.block_grid_w, batch.block_grid_h)

        latent = jax.vmap(reparameterize)(latent_mean, latent_log_variance, jax.random.split(key, latent_mean.shape[0]))

        scored = jax.vmap(score_teacher_layout, in_axes=(None, 0, 0, 0, 0, 0, 0))(params, latent, batch.block_grid_x, batch.block_grid_y, 
                                                                                  batch.block_grid_w,batch.block_grid_h, batch.footprint_option_index)

        kl_for_loss, kl_as_reported = jax.vmap(compute_kl_divergence_with_free_bits, in_axes=(0, 0, None))(latent_mean, latent_log_variance, free_bits)

        loss = (scored["cell_loss"].mean() + option_loss_weight * scored["option_loss"].mean() + kl_weight * kl_for_loss.mean())
        
        return loss, {"cell_loss": scored["cell_loss"].mean(),
                      "option_loss": scored["option_loss"].mean(),
                      "cell_accuracy": scored["cell_accuracy"].mean(),
                      "option_accuracy": scored["option_accuracy"].mean(),
                      "illegal_target_fraction": scored["illegal_target_fraction"].mean(),
                      "kl": kl_as_reported.mean()}

    @jax.jit
    def training_step(parameters, optimizer_state, batch, key):
        (loss, diagnostics), gradients = jax.value_and_grad(compute_loss, has_aux=True)(
            parameters, batch, key)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        diagnostics["loss"] = loss
        return parameters, optimizer_state, diagnostics

    return training_step


def build_parameter_evaluation_function(configuration, problem_instance):
    score_teacher_layout = build_teacher_forced_scoring_function(configuration, problem_instance)
    decode_latent_batch = build_greedy_decode_function(configuration, problem_instance)
    evaluate_floorplan_batch = build_evaluation_function(problem_instance)
    compute_objective = build_objective_function(configuration, problem_instance)
    free_bits = float(configuration["free_bits"])
    num_evaluation_layouts = int(configuration["num_evaluation_layouts"])

    encode_batch = jax.jit(jax.vmap(
        lambda parameters, x, y, w, h: encode_floorplan(parameters, configuration, problem_instance, x, y, w, h), in_axes=(None, 0, 0, 0, 0)))
    score_batch = jax.jit(jax.vmap(score_teacher_layout, in_axes=(None, 0, 0, 0, 0, 0, 0)))

    def summarise_decode(decoded):
        metrics = evaluate_floorplan_batch(decoded.block_grid_x, decoded.block_grid_y, decoded.block_grid_w, decoded.block_grid_h)
        hpwl = compute_original_hpwl_in_double_precision(metrics["net_grid_span_x"], metrics["net_grid_span_y"], problem_instance)
        return metrics, hpwl

    def evaluate_parameters(parameters, validation_set, key):
        subset = validation_set.take(jnp.arange(min(num_evaluation_layouts,
                                                    len(validation_set))))
        latent_mean, latent_log_variance = encode_batch(parameters, subset.block_grid_x, subset.block_grid_y, subset.block_grid_w, subset.block_grid_h)
        scored = score_batch(parameters, latent_mean, subset.block_grid_x, subset.block_grid_y, subset.block_grid_w, subset.block_grid_h, subset.footprint_option_index)
        
        _, kl_as_reported = jax.vmap(compute_kl_divergence_with_free_bits, in_axes=(0, 0, None))(latent_mean, latent_log_variance, free_bits)

        posterior_decoded = decode_latent_batch(parameters, latent_mean)
        posterior_metrics, posterior_hpwl = summarise_decode(posterior_decoded)

        prior_latent = jax.random.normal(key, latent_mean.shape)
        prior_decoded = decode_latent_batch(parameters, prior_latent)
        prior_metrics, prior_hpwl = summarise_decode(prior_decoded)


        prior_spread = 0.5 * float(prior_decoded.block_grid_x.astype(jnp.float32).std(axis=0).mean() + prior_decoded.block_grid_y.astype(jnp.float32).std(axis=0).mean())

        return {
            "cell_accuracy": float(scored["cell_accuracy"].mean()),
            "option_accuracy": float(scored["option_accuracy"].mean()),
            "illegal_target_fraction": float(scored["illegal_target_fraction"].mean()),
            "kl": float(kl_as_reported.mean()),
            "posterior_hpwl": float(posterior_hpwl.mean()),
            "posterior_alignment": float(posterior_metrics["alignment"].mean()),
            "posterior_overlap": float(posterior_metrics["overlap"].mean()),
            "prior_hpwl": float(prior_hpwl.mean()),
            "prior_hpwl_std": float(prior_hpwl.std()),
            "prior_alignment": float(prior_metrics["alignment"].mean()),
            "prior_overlap": float(prior_metrics["overlap"].mean()),
            "prior_spread": prior_spread,
            "prior_mean_objective": float(compute_objective(prior_metrics).mean()),
            "prior_best_objective": float(compute_objective(prior_metrics).max()),
            "num_infeasible_steps": int(prior_decoded.num_infeasible_steps.sum()),
            "num_footprint_retries": int(prior_decoded.num_footprint_retries.sum()),
        }

    return evaluate_parameters


def run_training(configuration, problem_instances, parameters, training_sets, validation_sets, start_step=0):
    output_directory = configuration["OUTPUT_DIRECTORY"]
    os.makedirs(output_directory, exist_ok=True)

    circuit_names = list(problem_instances)
    optimizer = create_optimizer(configuration)
    optimizer_state = optimizer.init(parameters)

    print(f"[compile] one training step and one evaluation per circuit, "
          f"{len(circuit_names)} circuits; each pays on its first use", flush=True)
    training_step_per_circuit = {
        name: build_training_step_function(configuration, instance, optimizer)
        for name, instance in problem_instances.items()}
    evaluate_parameters_per_circuit = {
        name: build_parameter_evaluation_function(configuration, instance)
        for name, instance in problem_instances.items()}

    batch_size = int(configuration["batch_size"])
    num_training_steps = int(configuration["num_training_steps"])
    evaluate_every = int(configuration["evaluate_every_num_steps"])
    log_every = int(configuration["log_every_num_steps"])

    batch_sampler = np.random.default_rng(int(configuration["training_seed"]))
    key = jax.random.PRNGKey(int(configuration["training_seed"]))

    evaluation_log = []
    started_at = time.time()

    for step in range(start_step, num_training_steps):
        circuit_name = circuit_names[batch_sampler.integers(0, len(circuit_names))]
        training_set = training_sets[circuit_name]
        batch = training_set.take(
            jnp.asarray(batch_sampler.integers(0, len(training_set), batch_size)))

        key, step_key = jax.random.split(key)
        parameters, optimizer_state, diagnostics = training_step_per_circuit[circuit_name](
            parameters, optimizer_state, batch, step_key)

        if step % log_every == 0:
            seconds_per_step = (time.time() - started_at) / (step - start_step + 1)
            print(f"[{step:6d}] {circuit_name:6s} loss={float(diagnostics['loss']):.4f} "
                  f"cell={float(diagnostics['cell_loss']):.4f} "
                  f"option={float(diagnostics['option_loss']):.4f} "
                  f"kl={float(diagnostics['kl']):.2f} "
                  f"acc={float(diagnostics['cell_accuracy']):.3f}/"
                  f"{float(diagnostics['option_accuracy']):.3f} "
                  f"illegal={float(diagnostics['illegal_target_fraction']):.4f} "
                  f"{seconds_per_step:.3f}s/step", flush=True)

        if (step + 1) % evaluate_every == 0 or step + 1 == num_training_steps:
            key, evaluation_key = jax.random.split(key)
            per_circuit = {}
            for name in circuit_names:
                evaluation_key, circuit_key = jax.random.split(evaluation_key)
                per_circuit[name] = evaluate_parameters_per_circuit[name](
                    parameters, validation_sets[name], circuit_key)

            record = {
                "step": step + 1,
                "seconds": time.time() - started_at,
                "mean_prior_objective": float(np.mean(
                    [per_circuit[name]["prior_mean_objective"] for name in circuit_names])),
                "mean_prior_overlap": float(np.mean(
                    [per_circuit[name]["prior_overlap"] for name in circuit_names])),
                "per_circuit": per_circuit,
            }
            evaluation_log.append(record)

            print(f"[eval {step + 1:6d}] mean prior J={record['mean_prior_objective']:+.4f}  "
                  f"mean prior overlap={record['mean_prior_overlap']:.4f}", flush=True)
            for name in circuit_names:
                c = per_circuit[name]
                print(f"{name:6s} J={c['prior_mean_objective']:+.4f} "
                      f"hpwl={c['prior_hpwl']:9.0f} align={c['prior_alignment']:.4f} "
                      f"ovl={c['prior_overlap']:.4f} cell_acc={c['cell_accuracy']:.3f} "
                      f"illegal={c['illegal_target_fraction']:.4f}", flush=True)

            save_parameters(parameters, os.path.join(output_directory, "last.npz"))
            save_parameters(parameters,
                            os.path.join(output_directory, f"snapshot_{step + 1:06d}.npz"))
            with open(os.path.join(output_directory, "evaluation_log.json"), "w") as log_file:
                json.dump(evaluation_log, log_file, indent=2)

    print(f"Done in {time.time() - started_at:.0f}s", flush=True)
    return parameters, evaluation_log
