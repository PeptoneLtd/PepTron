#!/bin/bash

export NCCL_TIMEOUT=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=.

# num_gpus = min(N, n_gpus_available) with N=len(CSV_FILE)
# max_batch_size=k*num_gpus with k positive integer and k<=N

CKPT_PATH="/mnt/data/checkpoints/peptron-stable-pdb-20250809/peptron-stable-pdb-20250809//dev/checkpoints/epoch=0-step=3699-consumed_samples=236800.0-last"
RESULTS_PATH="/mnt/data/this-is-a-test"
CSV_FILE="/mnt/data/datasets/splits/test-one.csv"

python -m peptron.infer \
    --config.inference.num_nodes 1 \
    --config.inference.checkpoint_path $CKPT_PATH \
    --config.inference.chains_path $CSV_FILE \
    --config.inference.results_path $RESULTS_PATH \
    --config.inference.num_gpus 1 \
    --config.inference.max_batch_size 1 \
    --config.inference.num_workers 8 \
    --config.inference.samples 10 \
    --config.inference.steps 10

python -m peptron.pt_to_structure -i "$RESULTS_PATH" \
    -o "$RESULTS_PATH/ensembles" \
    -p $(($(nproc) / 2))

# comment out the following lines only if you don't need to filter out unphysical conformations
mkdir -p "$RESULTS_PATH/physical_ensembles"
for trajectory_file in "$RESULTS_PATH/ensembles/"*.pdb; do
    [ -e "$trajectory_file" ] || continue
    base_name=$(basename "$trajectory_file" .pdb)
    output_file="$RESULTS_PATH/physical_ensembles/${base_name}_filtered.pdb"
    echo "Processing: $trajectory_file"
    python -m peptron.utils.filter_unphysical_traj --trajectory "$trajectory_file" --outfile "$output_file"
done
