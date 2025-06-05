# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.runner import run_replay_experiments
from src.types import APIModel, ReplayExperimentConfig


@hydra.main(
    version_base=None,
    config_path="config/human_replay",
    config_name="replay_human_games",
)
def main(cfg: ReplayExperimentConfig):
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

    run_replay_experiments(cfg)


if __name__ == "__main__":
    main()
