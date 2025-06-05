# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class BaselineModel:
    model_key: str
    model_id: str = field(default="", metadata={"help": "GloVe or Word2Vec"})
    global_guess: bool = field(
        default=False, metadata={"help": "Use global mode for decoder and interceptor"}
    )
    baseline_k: int = field(default=0, metadata={"help": "TopK selection parameter"})
    model_seed: int = field(
        default=0, metadata={"help": "Model generation random seed"}
    )


@dataclass
class LocalModel:
    model_key: str
    model_id: str = field(default="", metadata={"help": "Model id from HuggingFace"})
    urls: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of URLs for accessing the model, such as the URLs of the vllm servers"
        },
    )
    max_tokens: int = field(
        default=750, metadata={"help": "Maximum number of tokens for model outputs"}
    )
    temperature: float = field(
        default=0.6, metadata={"help": "Temperature for model outputs"}
    )
    model_seed: int = field(
        default=0, metadata={"help": "Model generation random seed"}
    )
    job_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "List of job IDs associated with the model"},
    )
    pass_system_as_user: bool = field(
        default=False,
        metadata={"help": "If True, system prompt is passed as user prompt (o1, R1)"},
    )


@dataclass
class APIModel:
    model_key: str
    model_id: str = field(
        default="",
        metadata={
            "help": "Model id to provide the API (e.g. 'claude-3-5-sonnet-20240620') "
        },
    )
    api_key_name: str = field(
        default="",
        metadata={"help": "Name of the env var corresponding to the API key"},
    )
    temperature: float = field(
        default=0.6, metadata={"help": "Temperature for model outputs"}
    )
    max_tokens: int = field(
        default=750, metadata={"help": "Maximum number of tokens for model outputs"}
    )
    model_seed: int = field(
        default=0,
        metadata={
            "help": "Model generation random seed. Not used by Anthropic models."
        },
    )
    pass_system_as_user: bool = field(
        default=False,
        metadata={"help": "If True, system prompt is passed as user prompt (o1, R1)"},
    )
    use_litellm: bool = field(
        default=False, metadata={"help": "Whether to use the LiteLLM API."}
    )
    num_retries: Optional[int] = field(
        default=None, metadata={"help": "Number of retries for LiteLLM API calls"}
    )
    provider_route: Optional[str] = field(
        default=None, metadata={"help": "Provider route for LiteLLM API calls"}
    )


@dataclass
class OpenAIModel(APIModel):
    api_host_name: str = field(
        default="",
        metadata={"help": "Name of the env var corresponding to the API host"},
    )
    use_azure: bool = field(
        default=False, metadata={"help": "Whether to use AzureOpenAI API"}
    )
    api_version: str = field(
        default="",
        metadata={"help": "OpenAI API version to use (e.g. '2024-12-01-preview')"},
    )
    reasoning_effort: str = field(
        default="low",
        metadata={
            "help": "Reasoning effort for OpenAI models. Can be 'low', 'medium', or 'high'."
        },
    )


@dataclass
class AnthropicModel(APIModel):
    max_reasoning_tokens: int = field(
        default=0,
        metadata={
            "help": "Maximum number of reasoning tokens for Extended Thinking. 0 disables ET. Only used by 3.7"
        },
    )


@dataclass
class Human:
    model_key: str = "human"
    model_id: Optional[str] = field(default="human")


@dataclass
class BaseExperimentConfig:
    # Basic experiment parameters
    num_episodes: int = field(
        default=2, metadata={"help": "Number of episodes to run for each experiment"}
    )
    env_seed: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "Environment random seed. Can provide multiple ones. Only first is used for HumanExperiments"
        },
    )
    baseline_data_dir: str = field(
        default="../embedding_models/",
        metadata={
            "help": "Baseline embeddings location. Downloaded automatically if not found."
        },
    )
    exp_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Experiment name. Results will be saved under `results/{exp_name}`"
        },
    )
    models: List[Union[BaselineModel, LocalModel, AnthropicModel, OpenAIModel]] = field(
        default_factory=list, metadata={"help": "List of models to run."}
    )
    confirm_include_api_models: bool = field(
        default=False,
        metadata={
            "help": "Confirm inclusion of API (paid) models. Used to prevent unwanted costs."
        },
    )
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    no_error_history: bool = field(
        default=False,
        metadata={"help": "Do not add error messages to the history (context)"},
    )
    model_seed: int = field(
        default=0,
        metadata={
            "help": "Model generation random seed. Used for models from slurm or all models if `--override_model_seed` is true."
        },
    )
    override_model_seed: bool = field(
        default=False, metadata={"help": "Whether to override the model seed."}
    )


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    # Flags restricting the model combinations
    fixed_encoder: str = field(
        default="", metadata={"help": "Key for a single, fixed encoder."}
    )
    fixed_decoder: str = field(
        default="", metadata={"help": "Key for a single, fixed decoder."}
    )
    fixed_interceptor: str = field(
        default="",
        metadata={
            "help": "Key for a single, fixed interceptor. Mainly used to study model cooperation."
        },
    )
    filter_model: str = field(
        default="",
        metadata={
            "help": "Key for a single model. Experiment will only run combinations including the filter model."
        },
    )
    match_encoder_decoder: bool = field(
        default=False,
        metadata={
            "help": "Whether to only look at combos where encoder == decoder. Mainly used to study model competition."
        },
    )
    match_encoder_decoder_models_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only look at combos where encoder_model == decoder_model. Mainly used to study effect of prompts (i.e. modes)."
        },
    )

    # Prompting parameters
    system_prompt_modes: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "List of system prompt modes to use"},
    )
    role_prompt_modes: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "List of role prompt modes to use"},
    )
    theory_of_mind: bool = field(
        default=False,
        metadata={
            "help": "Enable theory of mind prompt for the encoder. Not used in paper."
        },
    )
    tom_modes: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "List of tom modes to use. Only used if `theory_of_mind == True`."
        },
    )

    # Theory of Mind experiments, based on Piaget et al. (1956) and Gopnik & Astington (1988)
    piaget: bool = field(
        default=False,
        metadata={"help": "Whether to record Alice's prediction of Eve's guess."},
    )
    gopnik: bool = field(
        default=False,
        metadata={
            "help": "Whether to record Eve's ToM abilities according to Gopnik & Astington, 1988."
        },
    )

    # Alternative workflow: detect active LLM servers directly from the Slurm queue
    get_models_from_slurm: bool = field(
        default=False,
        metadata={
            "help": "Whether to detect and include local models directly from the Slurm queue"
        },
    )
    max_tokens: int = field(
        default=750,
        metadata={
            "help": "Maximum number of tokens for model outputs. Only used if getting models from slurm."
        },
    )
    temperature: float = field(
        default=0.6,
        metadata={
            "help": "Temperature for model outputs.  Only used if getting models from slurm."
        },
    )


@dataclass
class HumanExperimentConfig(BaseExperimentConfig):
    models: List[
        Union[BaselineModel, LocalModel, AnthropicModel, OpenAIModel, Human]
    ] = field(
        default_factory=list,
        metadata={
            "help": "List of models to evaluate, in (encoder, decoder, interceptor) order."
        },
    )

    def __post_init__(self):
        if len(self.models) != 3:
            raise ValueError(
                "The 'models' list must contain exactly 3 items (encoder, decoder, interceptor)."
            )


@dataclass
class ReplayExperimentConfig(ExperimentConfig):
    replay_data_dir: str = field(
        default="../results/human_data/human_data_pkl",
        metadata={"help": "Folder containing the history files for games to replay."},
    )
    replay_as_decoder: bool = field(
        default=False, metadata={"help": "Whether to replay games as a decoder."}
    )
    replay_as_interceptor: bool = field(
        default=False, metadata={"help": "Whether to replay games as an interceptor."}
    )

    def __post_init__(self):
        if not self.replay_as_decoder and not self.replay_as_interceptor:
            raise ValueError(
                "At least one of 'replay_as_decoder' or 'replay_as_interceptor' must be True."
            )
