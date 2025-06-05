# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess

agent_paths = {
    "llama3.1_8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1_70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepseek_r1_32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # DeepSeek R1 Distilled
}


def get_available_servers():
    # Run squeue and capture output
    result = subprocess.run(
        ["squeue", "--me", "-o", '"%j, %N, %T, %i"'], capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")

    # Initialize a list to store LocalModel instances
    local_models = []

    # Iterate over each line, skipping the header
    for line in lines[1:]:
        line = line.strip('"')
        # Get job name, nodelist, and status
        job_name, nodelist, status, job_id = line.split(", ")

        assert "[" not in nodelist, "Multi-node servers not currently supported."

        # Keep only running jobs
        if status == "RUNNING" and job_name != "bash":
            try:
                model_path = agent_paths[job_name]
                server_address = f"http://{nodelist}:8000/v1"

                # Check if a model with the same key already exists
                existing_model = next(
                    (m for m in local_models if m["model_key"] == job_name), None
                )

                if existing_model:
                    existing_model["urls"].append(server_address)
                    existing_model["job_ids"].append(job_id)
                else:
                    # Create a new LocalModel instance
                    local_model_info = {
                        "model_key": job_name,
                        "model_id": model_path,
                        "urls": [server_address],
                        "job_ids": [job_id],
                    }
                    local_models.append(local_model_info)
            except KeyError:
                continue

    return local_models


if __name__ == "__main__":
    running_jobs = get_available_servers()
    print(running_jobs)
