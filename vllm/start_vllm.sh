#!/usr/bin/env bash
set -euo pipefail

model_path="${1:?Usage: bash vllm/start_vllm.sh MODEL_DIR [extra vLLM arguments]}"
shift

export CUDA_VISIBLE_DEVICES="${VLLM_GPU:-${CUDA_VISIBLE_DEVICES:-1}}"

exec vllm serve "$model_path" \
    --host "${VLLM_HOST:-127.0.0.1}" \
    --port "${VLLM_PORT:-8001}" \
    --served-model-name "${VLLM_MODEL_NAME:-teacher}" \
    --dtype "${VLLM_DTYPE:-auto}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-3027}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-8}" \
    --max-logprobs -1 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
    --generation-config vllm \
    "$@"
