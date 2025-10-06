#!/usr/bin/env bash

set -euo pipefail

# Simple helper to generate train/validation/test splits with distinct seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

run_split() {
    local split_name="$1"
    local num_samples="$2"
    local seed="$3"
    local video_samples="$4"

    local output_h5="${ROOT_DIR}/data/random_dmp_${split_name}.h5"
    local video_dir="${ROOT_DIR}/videos/random_dmp_samples/${split_name}"

    echo "[nsf-rl] Generating ${split_name} split (${num_samples} samples, seed ${seed})"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_random_dmp_dataset.py" \
        --num-samples "${num_samples}" \
        --seed "${seed}" \
        --video-samples "${video_samples}" \
        --output-h5 "${output_h5}" \
        --video-dir "${video_dir}"
}

run_split train 10000 0 10
run_split validation 1000 1 6
run_split test 1000 2 6

echo "[nsf-rl] Dataset generation complete. Outputs available under data/ and videos/."
