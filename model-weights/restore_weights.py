# In order for the model weights to be uploaded to github the files have to be smaller than 25mb, which a fully trained model isn't.
# Luckly, for running inference the encoder is not necessary. So I remove the encoder weights from the checkpoints.
# The problem is that during inference the encoder weights do get loaded. This script is a hack, it simply adds zeros to the 
# encoder weights. They do nothing during inference so this is fine.
# 
# In order to add these zeros run the following:
#     python3 model-weights/restore_weights.py
#
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

MODEL_WEIGHTS_DIRECTORY = pathlib.Path(__file__).resolve().parent
STRIPPED_SUFFIX = ".stripped.npz"


def restore_one_checkpoint(stripped_path):
    exported = np.load(stripped_path)
    manifest = json.loads(str(exported["restore_manifest"]))
    assert manifest["format"] == 1, f"{stripped_path.name}: manifest format {manifest['format']}, this script writes 1"

    zero_filled_leaves = {int(index): (tuple(shape), dtype) for index, shape, dtype in manifest["zero_filled"]}

    payload = {}
    for index in range(int(manifest["total_leaves"])):
        key = f"parameter_leaf_{index:04d}"
        if index in zero_filled_leaves:
            shape, dtype = zero_filled_leaves[index]
            payload[key] = np.zeros(shape, dtype=np.dtype(dtype))
        else:
            assert key in exported.files, f"{stripped_path.name}: {key} is neither stored nor listed as zero-filled"
            payload[key] = exported[key]


    restored_path = stripped_path.with_name(stripped_path.name[:-len(STRIPPED_SUFFIX)] + ".npz")
    temporary_path = str(restored_path) + ".partial.npz"
    np.savez(temporary_path, **payload)
    os.replace(temporary_path, restored_path)

    print(f"{stripped_path.name:<38} -> {restored_path.name:<28} "
          f"{restored_path.stat().st_size / 1e6:6.2f} MB  "
          f"({len(payload) - len(zero_filled_leaves)} restored, "
          f"{len(zero_filled_leaves)} zero-filled)")
    return restored_path


def main():
    stripped_paths = sorted(MODEL_WEIGHTS_DIRECTORY.glob("*" + STRIPPED_SUFFIX))
    assert stripped_paths, f"no *{STRIPPED_SUFFIX} files in {MODEL_WEIGHTS_DIRECTORY}"

    for stripped_path in stripped_paths:
        restore_one_checkpoint(stripped_path)


if __name__ == "__main__":
    main()
