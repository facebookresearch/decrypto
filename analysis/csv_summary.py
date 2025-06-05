# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re

import pandas as pd


def summarize_experiments(results_dir):
    # List all directories in the results directory
    experiment_folders = sorted(
        [
            f
            for f in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, f))
        ]
    )

    # Prepare DataFrame to collect all results
    columns = [
        "encoder",
        "encoder_mode",
        "decoder",
        "decoder_mode",
        "interceptor",
        "interceptor_mode",
        "intercepts",
        "miscomms",
        "both",
        "survived",
    ]
    all_rows = []

    if "ToM" in experiment_folders[0]:
        columns.append("encoder_tom_mode")

    # Process each experiment folder
    for folder in experiment_folders:
        # Extract model and mode information from folder name
        parts = re.split("-|_", folder)

        if len(parts) < 7:
            continue  # Skip if folder name does not match expected format

        encoder = "_".join(parts[0:2])
        decoder = "_".join(parts[2:4])
        interceptor = "_".join(parts[4:6])
        modes = parts[6]

        if "ToM" in folder:
            encoder_tom_mode = int(folder.split("ToM")[-1])

        # Read stats.json file
        stats_path = os.path.join(results_dir, folder, "stats.json")

        if not os.path.exists(stats_path):
            continue  # Skip if stats file does not exist

        with open(stats_path, "r") as file:
            stats = json.load(file)

        # Append data to list
        row = {
            "encoder": encoder,
            "encoder_mode": modes[0],
            "decoder": decoder,
            "decoder_mode": modes[1],
            "interceptor": interceptor,
            "interceptor_mode": modes[2],
            "intercepts": stats.get("intercept", 0),
            "miscomms": stats.get("miscomm", 0),
            "both": stats.get("both", 0),
            "survived": stats.get("survived", 0),
        }
        if "ToM" in folder:
            row["encoder_tom_mode"] = encoder_tom_mode
        all_rows.append(row)

        print(folder)

    # Convert list to DataFrame
    df = pd.DataFrame(all_rows, columns=columns)
    # Save DataFrame to CSV
    output_path = os.path.join(results_dir, "summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSummary CSV has been saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize experiment results into a CSV file."
    )
    parser.add_argument(
        "--results_dir", type=str, help="Directory containing experiment results."
    )
    args = parser.parse_args()

    summarize_experiments(args.results_dir)
