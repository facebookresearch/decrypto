<p align="center">
  <a href="https://arxiv.org/abs/2506.20664">
    <img src="assets/Decrypto Title Thin.png" width="55%" alt="Decrypto Logo" />
  </a>
</p>

---

<div align="center">

<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-lightgrey"/></a>
  <a href="https://sites.google.com/view/decrypto-ai"><img src="https://img.shields.io/badge/blog-Decrypto-%230a7c9d?style=flat"/></a>
  <a href="https://arxiv.org/abs/2506.20664"><img src="https://img.shields.io/badge/arXiv-2506.22419-b31b1b.svg"/></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
</p>

<img src="assets/Decrypto Game Diagram.png" alt="Decrypto Game Diagram" width="80%">

[**Installation**](#install) | [**Quick Start**](#start) | [**Structure**](#structure) | [**Citation**](#cite)
---
</div>


Decrypto is a benchmark for multi-agent reasoning and theory of mind in language models. It is based on the board game of the same name. 

The first purpose of Decrypto is to evaluate the cooperation and competition capabilities of LLMs in a simple word-guessing game.
The second purpose is to provide a framework to study Theory of Mind in LLMs, and we implement multiple experiments inspired by seminal works in the field of cognitive science.


## Features 🦾

- Evaluates cooperation and competition in LLMs
- Provides an easy-to-extend framework to study Theory of Mind
- Supports both cloud-based and local LLMs
- Includes competitive word embedding (i.e. non-LLM) baseline agents
- Provides the tools to play the game with LLMs and collect human data

<h2 name="install" id="install">Installation ⚙️ </h2>

Create a new virtual environment and install the requirements:
```shell
conda create -n decrypto python=3.10
conda activate decrypto
git clone https://github.com/your-repo/decrypto.git
cd decrypto
pip install -r requirements.txt
```

### Setup API keys
To use API models, create a `.env` file under the project root directory to store your API keys. Ex:
```shell
echo "ANTHROPIC_API_KEY=your-anthropic-api-key-here" >> .env
echo "OPENAI_PRIMARY_KEY=your-openai-api-key-here" >> .env
```

These keys will be automatically loaded when running experiments. 
Ensure `.env` is listed in `.gitignore` to avoid accidentally exposing your keys.


<h2 name="start" id="start">Quick Start 🚀 </h2>

### Play with LLMs 👾

_The best way to understand what Decrypto is about is to play the game yourself!_

After setting up the API key, you can play with GPT-4o using:
```shell
python human_play.py --config-path=config/human_play --config-name=play_with_gpt4o
```

To play with Llama 3.1-70B-Instruct, first use [vLLM](https://docs.vllm.ai/en/latest/) to spin up a local instance of the model.
```shell
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --enable-prefix-caching --tensor-parallel-size 8
```
If this is your first time using vLLM on that machine, you will have to login to your Hugging Face account. Follow the instructions [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login).
You will also have to accept the Llama 3.1 [license agreement](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct).

Once the instance booted up, run the game in a separate terminal with:
```shell
python human_play.py --config-path=config/human_play --config-name=play_with_llama
```
The first game prompt will provide the rules, and may require scrolling on smaller screens. You will play as the Encoder. Try to make it through 8 rounds without miscommunicating with the Decoder or being intercepted by the Interceptor.

### Running experiments 🔭

We use [Hydra](https://hydra.cc/) to manage configs and easily launch up experiments. 
To run an experiment, use `run.py` with one of the config files in `config/`:
```shell
python run.py --config-path={config_path} --config-name={config_name}
```

For example, to run a game with either a local instance of Llama 3.1-70B-Instruct or with GPT-4o, use one of the following commands:
```shell
# Run with Llama 3.1-70B-Instruct
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --enable-prefix-caching --tensor-parallel-size 8
python run.py --config-path=config/examples --config-name=local

# Run with GPT-4o, after setting up the API key
python run.py --config-path=config/examples --config-name=api_gpt4o
```

To create a HTML version of a game transcript, use the `chat_visualizer.py` script:
```shell
python analysis/chat_visualizer.py --experiment_dir results/local_example --model_combo llama3.1_70B_llama3.1_70B_llama3.1_70B_000_000 --output_name llama3.1_70B.html
# or
python analysis/chat_visualizer.py --experiment_dir results/api_gpt4_example --model_combo gpt-4o_gpt-4o_gpt-4o_000_000 --output_name gpt-4o.html
```
Transcripts are saved in `game_transcripts/` and can be opened in any web browser.

<h2 name="structure" id="structure">Repo Structure ⛩️️ </h2>

### Overview

There are three entry points to the code:
1. `run.py`: The main entry point to run experiments.
2. `human_play.py`: The entry point to play the game with LLMs. It runs the game in the console and allows any combination of human and AI players.
3. `human_replay.py`: The entry point to replay saved human games recorded with `human_play.py`. Used to evaluate other LLMs on human data.
 
All three use Hydra to manage configs and easily launch up experiments.

### Human Data

Data from the 10 games played by humans against Llama 3.1-70B-Instruct are provided in `results/human_data/`, in both JSON and Pickle format.
Those are the games used to compute Table 1 in the paper.

### Setting up configs

Configs are stored in `config/` and follow a similar structure for all entry points. 

Models are provided as a list, with generation parameters being set individually for each model. 
By default, `run.py` will run games for all combinations of 3 models (Encoder, Decoder, Interceptor) in the list, computing a `N x N x N` matrix of results.
Since that can be quite expensive, we implement a set of flags like `match_encoder_decoder` or `fix_interceptor` to restrict experiments to a subset of combinations and only compute a slice of the full matrix.

Flags for experiments and models are defined in `src/types.py`.

### Using Slurm
If using a cluster with Slurm, we provide a set of scripts in `slurm/` to host and monitor vLLM instances on the cluster in a way that makes them accessible to the rest of the code.

### Reproducing results
To reproduce our results, we provide a set of config files for that purpose in `config/paper`.


<h2 name="contrib" id="cite">Contribute 🛠️</h2>
We welcome contributions to Decrypto, including new theory of mind experiments, new agents, and improvements to the codebase.

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

<h2 name="cite" id="cite">Citing Decrypto 📜 </h2>
If you use Decrypto in your research, please cite the following paper:

```bibtex
@article{lupu2025decrypto,
  title={The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind}, 
  author={Andrei Lupu and Timon Willi and Jakob Foerster},
  year={2025},
  eprint={2506.20664},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.20664}, 
}
```

## License 
Decrypto is CC-BY-NC licensed, as found in the LICENSE file.