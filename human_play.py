# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl
from datetime import datetime

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.runner import run_games_with_humans
from src.types import APIModel, HumanExperimentConfig


@hydra.main(
    version_base=None, config_path="config/human_play", config_name="play_with_llama"
)
def main(cfg: HumanExperimentConfig):
    print(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, "override_model_seed") and cfg.override_model_seed:
        models = [instantiate(m, model_seed=cfg.model_seed) for m in cfg.models]
    else:
        models = [instantiate(m) for m in cfg.models]

    cfg = instantiate(cfg)
    cfg.models = models

    for model in cfg.models:
        if isinstance(model, APIModel):
            assert cfg.confirm_include_api_models, (
                "Need `confirm_include_api_models=True` to approve using API models. This prevents unwanted costs."
            )

    cfg.env_seed = cfg.env_seed[0]

    # Run the experiment. Note: prompt modes are hardcoded to 0 for now.
    results = run_games_with_humans(
        encoder_policy=(cfg.models[0], 0, 0, 0),
        decoder_policy=(cfg.models[1], 0, 0),
        interceptor_policy=(cfg.models[2], 0, 0),
        args=cfg,
    )

    os.makedirs("results/human_data/", exist_ok=True)
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%y")
    pkl.dump(
        results, open(f"results/human_data/{cfg.exp_name}_{current_time}.pkl", "wb")
    )

    print(f"Games data saved in results/human_data/{cfg.exp_name}_{current_time}.pkl")


if __name__ == "__main__":
    main()
