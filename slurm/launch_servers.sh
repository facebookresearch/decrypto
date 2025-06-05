# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

models=(
#     "llama3.1_8B:meta-llama/Meta-Llama-3.1-8B-Instruct"
#     "llama3.1_70B:meta-llama/Meta-Llama-3.1-70B-Instruct"
#     "deepseek_r1_32B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

for model in "${models[@]}"; do
  # Extract the model key and name
  model_key=$(echo $model | cut -d ':' -f 1)
  model_name=$(echo $model | cut -d ':' -f 2)

  # Submit server job to SLURM
  sbatch --gpus-per-node=8 --time=24:00:00 --cpus-per-task 96 --nodes 1 --output "/slurm_logs/vllm/${model_key}-%j.out" \
  --error "/slurm_logs/vllm/${model_key}-%j.err" --job-name $model_key --mem 0 --exclusive \
  --wrap "vllm serve $model_name --enable-prefix-caching --tensor-parallel-size 8 --trust-remote-code --disable-sliding-window --disable-log-stats"
done