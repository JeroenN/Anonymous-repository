from __future__ import annotations

# References values estimated from FlexPlanner performance.
HPWL_REFERENCE_PER_CIRCUIT = {
    "ami33": 60_000.0, 
    "ami49": 780_000.0,
    "n10": 30_000.0, 
    "n30": 85_000.0, 
    "n50": 110_000.0,
    "n100": 180_000.0, 
    "n200": 320_000.0, 
    "n300": 460_000.0
}

# Objective function adapted from FlexPlanner
def build_objective_function(configuration, problem_instance):
    hpwl_reference = HPWL_REFERENCE_PER_CIRCUIT[problem_instance.circuit_name]
    alignment_weight = float(configuration["objective_alignment_weight"])
    overlap_weight = float(configuration["objective_overlap_weight"])

    def compute_objective(metrics):
        return (1.0 - metrics["hpwl"] / hpwl_reference + alignment_weight * metrics["alignment"] - overlap_weight * metrics["overlap"])

    compute_objective.description = (f"J = 1 - HPWL/{hpwl_reference:,.0f} + {alignment_weight}*alignment "
                                     f"- {overlap_weight}*overlap")
    return compute_objective
