  #!/bin/bash

export NCCL_TIMEOUT=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=.

# num_gpus = min(N, n_gpus_available) with N=len(CSV_FILE)
# max_batch_size=k*num_gpus with k positive integer and k<=N

CHECKPOINT_PATH="/path/to/the/peptron-checkpoint"
RESULTS_PATH="results"
INPUT_CSV=""
FILTER_UNPHYSICAL=false

OPTS=$(getopt \
    --longoptions "input:,checkpoint:,results:,filter-unphysical" \
    --options "i:c:r:f" \
    --name "$(basename "$0")" \
    -- "$@")

if [ $? -ne 0 ]; then
    echo "Failed to parse options" >&2
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        -i|--input)
            INPUT_CSV="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -r|--results)
            RESULTS_PATH="$2"
            shift 2
            ;;
        -f|--filter-unphysical)
            FILTER_UNPHYSICAL=true
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_CSV" ]; then
    echo "Error: --input is required" >&2
    exit 1
fi

python -m peptron.infer \
    --config.inference.num_nodes 1 \
    --config.inference.checkpoint_path $CHECKPOINT_PATH \
    --config.inference.chains_path $INPUT_CSV \
    --config.inference.results_path $RESULTS_PATH \
    --config.inference.num_gpus 1 \
    --config.inference.pipeline_model_parallel_size 1 \
    --config.inference.tensor_model_parallel_size 1 \
    --config.inference.micro_batch_size 1 \
    --config.inference.max_batch_size 48 \
    --config.inference.num_workers 8 \
    --config.inference.samples 10 \
    --config.inference.steps 10 \
    --config.inference.tmax 1

for d in $RESULTS_PATH/*/; do
  python -m peptron.compress_ensemble --pdb-dir "$d" --filter-unphysical
done
