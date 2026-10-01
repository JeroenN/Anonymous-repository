# Latent Geometric Optimization for 3D Floorplanning

A variational autoencoder that places blocks onto two stacked dies, written in pure JAX.
The project does two things: **train a model** (`train.py`) and **run inference with a
trained model** (`infer.py`). Both scripts take as input what config file should be used.



## The trained models

| file | trained on |
|---|---|
| `model-weights/8-circuit-generalist.npz` | all eight benchmark instances |
| `model-weights/N100-specialist.npz` | the `n100` instance only |

**Both models can run inference on all eight circuits.** Neither is restricted to the
instance it was trained on: a block's identity is computed from its static geometric and
netlist properties rather than looked up in a per-circuit embedding table, so the same
weights apply to any instance regardless of its block count.


## Installation

The only system requirement is a CUDA 12 driver. The CUDA libraries themselves come in
through pip.

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running inference

There are two ready-made configs, one per model:

```bash
source /path/to/venv/bin/activate

# the n100 specialist
python3 infer.py --config configs/infer_n100.yaml

# the 8-circuit generalist
python3 infer.py --config configs/infer_generalist8.yaml


```

### What a run produces

Written to `OUTPUT_DIRECTORY` (`output/` by default):

| file | contents |
|---|---|
| `inference_results.json` | per circuit and per seed: objective, HPWL, alignment, overlap, and the search history |
| `best_layout_<circuit>_seed<n>.npz` | the winning layout as `block_grid_x/y/w/h` |
| `compilation-cache/` | compiled XLA kernels, reused by later runs |

The JSON is rewritten after every circuit, so partial results survive an interrupted run.
The compilation cache makes the first run of a given circuit noticeably slower than the
ones after it; it is safe to delete.


## Config fields worth knowing

The config is the only input, the following fiels are important:

| field | effect |
|---|---|
| `CHECKPOINT_PATH` | which trained model to search with |
| `CIRCUIT_NAMES` | which instances to run; drop entries to run a subset |
| `search_seeds` | a list of seeds, each searched independently. |
| `search_seed` | the single seed used when `search_seeds` is empty |
| `search_population` | candidate layouts decoded per round |
| `search_rounds` | search iterations per circuit |
| `OUTPUT_DIRECTORY` | where results go |


## Training

`train.py` takes a config the same way:

```bash
python3 train.py --config configs/train_generalist8.yaml
```

The two training configs need `TEACHER_DATASET_PATHS` filled in before they will run.