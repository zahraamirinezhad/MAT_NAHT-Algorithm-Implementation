# MAT-NAHT with Visibility-Constrained Multi-Agent Coordination

**A Transformer-based Multi-Agent Reinforcement Learning (MARL) implementation for N-Agent Ad Hoc Teamwork under partial observability and restricted inter-agent visibility.**

This project implements a MAT-NAHT-style sequence-modeling approach within an NAHT/EPyMARL research codebase, using POAM as a key baseline and experimental framework. It explores how controlled agents can use recent observation–action histories and a visibility/communication mask to coordinate when they cannot freely observe every teammate.

> **Research attribution:** This repository builds on the open-source NAHT/EPyMARL codebase by Caroline Wang and collaborators. The MAT-NAHT components and visibility-mask functionality in this archive should be understood as a project implementation/extension, not as an official release from the original paper authors.

## Table of Contents

- [Research Overview](#research-overview)
- [Research Questions](#research-questions)
- [Related Papers](#related-papers)
- [Method Overview](#method-overview)
- [Visibility and Partial Observability](#visibility-and-partial-observability)
- [Implementation Details](#implementation-details)
- [Environments and Configurations](#environments-and-configurations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Configuration Reference](#configuration-reference)
- [Logging and Evaluation](#logging-and-evaluation)
- [Reproducibility and Current Limitations](#reproducibility-and-current-limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Research Overview

### The problem: N-Agent Ad Hoc Teamwork

In **N-Agent Ad Hoc Teamwork (NAHT)**, a controlled group of agents must cooperate with teammates whose identities, policies, or team composition may be unknown or may change over time. Unlike conventional cooperative MARL, the learning system may not control every agent, and cannot assume that all teammates were trained together.

This becomes more challenging under **partial observability**. An agent may receive only a local observation and may not be able to see or infer the state and behavior of every other agent. Effective coordination must therefore account for incomplete information and the history of interaction.

### Project objective

This project investigates a sequence-modeling approach inspired by **Sequence Modeling for N-Agent Ad Hoc Teamwork (MAT-NAHT)** in the NAHT/POAM experimental setting, with explicit attention to visibility constraints.

The implementation combines:

- A Transformer encoder over recent per-agent observation and action histories
- A Transformer decoder for action-logit generation and autoregressive action selection
- A centralized value-estimation component
- A communication/visibility mask that can restrict cross-agent attention
- An actor–critic learning pipeline and configurable training/evaluation experiments
- Open-team experiment configurations that support controlled and uncontrolled teammates

The goal is to study whether sequence-based representations of recent team behavior can support coordination when agents have limited access to information about one another.

## Research Questions

The project is organized around the following questions:

1. **Sequence modeling:** Can a Transformer over recent observation–action histories represent useful inter-agent dynamics for ad hoc coordination?
2. **Restricted visibility:** How does restricting which agents can attend to one another affect policy learning and team performance?
3. **Generalization:** Can a learned policy coordinate with varying or previously unseen teammate policies in open-team settings?
4. **Baseline comparison:** How does the sequence-modeling implementation compare with POAM under matched environments, training budgets, seeds, and evaluation protocols?

These are research questions the code can be used to investigate; they are not claims of an experimental result by themselves. Report conclusions only after running and comparing the corresponding experiments.

## Related Papers

### 1. Sequence Modeling for N-Agent Ad Hoc Teamwork — MAT-NAHT

**This is the primary paper motivating the sequence-modeling approach.**

- **Authors:** Caroline Wang, Di Yang Shi, Elad Liebman, Ishan Durugkar, Arrasy Rahman, and Peter Stone
- **Publication:** arXiv preprint, 2025
- **arXiv identifier:** 2506.05527
- **Research area:** Multi-agent reinforcement learning, sequence modeling, ad hoc teamwork
- **Links:** [arXiv abstract](https://arxiv.org/abs/2506.05527) · [PDF](https://arxiv.org/pdf/2506.05527)

**Main idea:** The paper proposes a centralized Transformer-based approach that uses historical observations and actions of controlled agents to model team interaction. It targets NAHT scenarios with varying numbers and types of teammates, including partially observable environments.

**Connection to this repository:** The implementation in `src/components/mat_naht/mat_naht.py` includes multi-head attention, Transformer encoder/decoder blocks, observation–action history processing, and a communication mask. The agent and critic implementations are in `src/modules/agents/mat_naht_agent.py` and `src/modules/critics/mat_naht.py`.

**Important distinction:** The visibility-mask extension and the exact experimental configuration in this repository should be treated as this project's implementation choices. Do not assume that every visibility setting, masking behavior, or reported result is part of the original paper unless it is verified against the paper and its official code.

### 2. N-Agent Ad Hoc Teamwork — POAM

**This is the foundational paper for the NAHT problem formulation and the POAM baseline.**

- **Authors:** Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone
- **Venue:** Advances in Neural Information Processing Systems (NeurIPS 2024), Main Conference Track
- **Volume:** *Advances in Neural Information Processing Systems 37*
- **DOI:** [10.52202/079017-3551](https://doi.org/10.52202/079017-3551)
- **Links:** [Official NeurIPS paper page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cabf611498431ad89a85ace75f790d93-Abstract-Conference.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cabf611498431ad89a85ace75f790d93-Paper-Conference.pdf) · [Official authors' code repository](https://github.com/carolinewang01/naht)

**Main idea:** The paper formalizes N-agent ad hoc teamwork and introduces **Policy Optimization with Agent Modelling (POAM)**, a policy-gradient approach that learns representations of teammate behavior to adapt to changing team composition and unseen teammates.

**Connection to this repository:** The repository contains POAM-specific agents, learners, critics, configurations, and open-team training/evaluation utilities. POAM provides a useful reference point for evaluating the sequence-modeling approach.

### How the two papers relate to this project

| Component | Research connection |
| --- | --- |
| NAHT problem setup and open-team evaluation | Wang et al., NeurIPS 2024 |
| POAM baseline and agent-modelling training pipeline | Wang et al., NeurIPS 2024 |
| Transformer-based modeling of observation–action histories | Wang et al., *Sequence Modeling for N-Agent Ad Hoc Teamwork*, arXiv 2506.05527 |
| Explicit visibility/communication masking | Implementation extension in this repository; verify the exact behavior and settings in code |
| Visibility-constrained comparison against POAM | Proposed experimental study; results must be generated and reported separately |

## Method Overview

At a high level, the MAT-NAHT agent processes a fixed-length history of observations and actions for each agent, uses attention to form representations, and produces action logits. A value head provides value estimates for actor–critic learning.

```text
Recent observations and actions
              │
              ▼
   Fixed-length history buffer
              │
              ▼
   Observation/action embeddings
              │
              ▼
 Transformer encoder with visibility mask
              │
              ▼
      Agent representations
              │
       ┌──────┴────────┐
       ▼               ▼
 Transformer       Centralized value
 decoder            estimation
       │               │
       ▼               ▼
 Action logits     Value targets
       │
       ▼
 Available-action masking / action selection
       │
       ▼
 Multi-agent environment
```

The diagram is a conceptual summary of the code, not a guarantee that every training and rollout path uses identical tensor shapes or exactly the same mask at every stage.

### Main components

- **History representation:** Recent observations and actions are assembled into a fixed-length window. Early timesteps are zero-padded.
- **Transformer encoder:** Projects observation–action features into a hidden representation and applies multi-head attention.
- **Transformer decoder:** Uses agent representations and previous/sample actions to produce action logits. Rollout code can select actions autoregressively.
- **Value estimation:** The MAT-NAHT critic uses the learned representation to estimate per-agent values for actor–critic optimization.
- **Policy optimization:** The MAT-NAHT configuration uses an on-policy PPO-style learning setup, including clipped policy updates, advantage estimation, and configurable value-loss options.
- **Open-team training/evaluation:** Configurations can load trained agents and populate an environment with uncontrolled teammates.

## Visibility and Partial Observability

Visibility is a central aspect of this project. A visibility or communication matrix represents which agent pairs may exchange information through attention.

Let \(M_{ij}\) denote whether agent \(i\) may attend to agent \(j\):

- \(M_{ij}=1\): the attention connection is permitted.
- \(M_{ij}=0\): the corresponding attention scores are masked before softmax.
- Self-visibility is preserved by the visibility-mask utility.

The mask is passed into the Transformer encoder and decoder attention mechanisms, allowing the model to restrict information flow between agents rather than assuming that all agents can always observe one another.

### Visibility-mask behavior in the supplied code

The utility `build_communication_mask` in `src/components/mat_naht/mat_naht.py` works as follows:

1. If the episode batch includes `neighbor_mask`, that mask is used directly.
2. Otherwise, a mask is generated by `extract_visibility_mask`.
3. The current default generator uses a random visibility gate with `availability_prob=0.7`.
4. The diagonal is forced to remain visible, preserving self-attention.
5. Optional distance-based masking and random occlusion are supported by the utility function, but the current call in `MATNAHTAgent` does not pass those parameters explicitly. Therefore, the call uses the function defaults unless the code is updated.

**Reproducibility note:** The default random visibility mask is sampled inside the agent's forward path. If you want a controlled visibility experiment, make the visibility probability, distance threshold, symmetry, and occlusion probability explicit configuration values; seed the mask generator; and decide whether masks should remain fixed for an episode or be resampled at each forward pass. This distinction can materially affect the experiment.

### Recommended visibility ablations

To evaluate the contribution of visibility constraints, consider comparing:

| Setting | Suggested interpretation |
| --- | --- |
| Full visibility | All agent-to-agent connections are enabled |
| Random visibility | A controlled fraction of connections is available |
| Distance-based visibility | Connections depend on a spatial-distance threshold, where the observation representation supports meaningful positions |
| Random occlusion | Additional visibility edges are randomly removed |
| Symmetric visibility | If agent \(i\) sees \(j\), agent \(j\) also sees \(i\) |
| POAM baseline | Baseline under the same environment and evaluation protocol |

These are suggested experiment settings, not results already included in the uploaded archive. Ensure the selected visibility mode is actually wired into the agent before reporting it.

## Implementation Details

| Item | Current configuration / location |
| --- | --- |
| Algorithm configuration | `src/config/algs/sc2/mat_naht.yaml` and `src/config/algs/mpe/mat_naht.yaml` |
| POAM baseline configuration | `src/config/algs/sc2/poam.yaml` and `src/config/algs/mpe/poam.yaml` |
| Agent | `src/modules/agents/mat_naht_agent.py` |
| Transformer and visibility utilities | `src/components/mat_naht/mat_naht.py` |
| Critic | `src/modules/critics/mat_naht.py` |
| Learner | `src/learners/mat_naht_learner.py` |
| Training entry point | `src/main.py` |
| Environment configs | `src/config/envs/` |
| Open-team configs | `src/config/open/` |

### Default MAT-NAHT hyperparameters

The provided `sc2/mat_naht.yaml` and `mpe/mat_naht.yaml` define the following main settings:

| Parameter | Default in MAT-NAHT config |
| --- | ---: |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| History length | 5 |
| Dropout | 0.1 |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.0003 |
| Discount factor \(\gamma\) | 0.99 |
| GAE \(\lambda\) | 0.95 |
| PPO clip parameter | 0.2 |
| PPO epochs | 5 |
| Minibatches | 2 |
| Entropy coefficient | 0.01 |
| Gradient norm clipping | 0.5 |
| Batch size | 64 |
| Parallel environments per run | 8 |

These are configuration values, not guaranteed optimal settings. Confirm any command-line overrides before comparing experiments.

## Environments and Configurations

The repository includes configuration files for two broad environment families:

### StarCraft II / SMAC

The SC2 configuration files include maps such as:

- `5m_vs_6m`
- `8m_vs_9m`
- `10m_vs_11m`
- `3s_vs_5z`

The map names and available scenarios depend on the installed StarCraft II version and SMAC maps.

### Multi-Agent Particle Environments (MPE)

The repository also includes open-team configurations for a predator–prey scenario through the custom `multiagent-particle-envs` dependency, including `mpe:PredatorPrey-v0`.

Use an environment and map supported by the installed dependencies. The configuration files are experiment templates; they do not imply that every environment has already been trained or evaluated successfully.

## Repository Structure

```text
.
├── requirements.txt
├── env_requirements.txt
├── install_sc2.sh
├── train_agent.sh
├── train_naht.sh
├── run_test.sh
└── src/
    ├── main.py
    ├── run.py
    ├── run_eval.py
    ├── components/
    │   ├── episode_buffer.py
    │   └── mat_naht/
    │       └── mat_naht.py
    ├── config/
    │   ├── algs/
    │   │   ├── sc2/
    │   │   └── mpe/
    │   ├── envs/
    │   └── open/
    │       └── mat_naht/
    ├── controllers/
    ├── envs/
    ├── learners/
    │   ├── mat_naht_learner.py
    │   └── poam_learner.py
    ├── modules/
    │   ├── agents/
    │   │   └── mat_naht_agent.py
    │   ├── agent_loaders/
    │   └── critics/
    │       └── mat_naht.py
    ├── runners/
    └── utils/
```

## Installation

The repository is based on a research-oriented PyMARL/EPyMARL setup and includes dependencies for PyTorch-based learning, Sacred experiment tracking, SMAC/StarCraft II, and custom multi-agent environments.

### Requirements

- Linux is recommended for the StarCraft II / SMAC workflow.
- Python version compatible with the pinned and legacy environment packages.
- PyTorch and a compatible CUDA setup if GPU training is desired.
- StarCraft II and SMAC maps for SC2 experiments.
- Git access to the custom environment repositories listed in `env_requirements.txt`.

The supplied dependencies include older packages such as `gym==0.23.1`, `pysc2==3.0.0`, and `PyYAML==5.3.1`. These may require an older Python environment and can conflict with modern packages. A dedicated Conda or virtual environment is recommended.

### 1. Clone the repository

After publishing the project to your own GitHub account, replace the placeholder URL:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create an isolated environment

Example using Conda:

```bash
conda create -n mat-naht python=3.8
conda activate mat-naht
```

Python 3.8 is a starting point for this legacy dependency stack, not a guarantee that every dependency will install on every platform. If installation fails, check the compatibility requirements of the exact PyTorch, SMAC, PySC2, and custom environment versions.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r env_requirements.txt
```

The environment requirements install custom Git repositories. If dependency resolution fails, install the compatible versions individually and record the final working environment in a pinned requirements or environment file.

### 4. Install StarCraft II and SMAC maps

For the repository's expected directory layout, review `install_sc2.sh` before running it. The script downloads StarCraft II and SMAC maps under a `pymarl/3rdparty` directory and assumes a specific project location.

Set `EXP_DIR` to the parent directory expected by the script, and ensure the script is compatible with the current download URLs and your system before execution. Do not run the script blindly on a machine where its paths or legacy downloads are unsuitable.

## Running Experiments

The framework uses configuration-driven commands through `src/main.py`. Commands below reflect the configuration names in the supplied archive; training itself has not been verified as part of preparing this README.

### Train MAT-NAHT on StarCraft II `5m_vs_6m`

```bash
python src/main.py \
  --env-config=sc2 \
  --config=open/mat_naht/open_train_5v6_mat_naht \
  --alg-config=sc2/mat_naht \
  with env_args.map_name=5m_vs_6m \
  --seed=112358
```

### Train POAM as a baseline on `5m_vs_6m`

```bash
python src/main.py \
  --env-config=sc2 \
  --config=open/open_train_5v6 \
  --alg-config=sc2/poam \
  with env_args.map_name=5m_vs_6m \
  --seed=112358
```

### Run the MAT-NAHT configuration on Predator–Prey (MPE)

```bash
python src/main.py \
  --config=open/mat_naht/open_train_pp_mat_naht \
  --alg-config=mpe/mat_naht \
  --env-config=mpe \
  with t_max=3000000 \
  env_args.key="mpe:PredatorPrey-v0" \
  env_args.time_limit=200 \
  env_args.pretrained_wrapper="PretrainedTag" \
  --seed=112358
```

Check that the custom MPE environment and pretrained wrapper are installed before running this command.

### Important note about the supplied shell scripts

`train_naht.sh` currently launches the **POAM** configuration (`--alg-config=sc2/poam`) despite its filename. Review the algorithm configuration in any shell script before interpreting the run as MAT-NAHT. The `run_test.sh` file contains commented example commands and is not a ready-to-run automated test suite.

## Logging and Evaluation

The framework uses Sacred-based experiment logging and supports TensorBoard according to configuration settings. Depending on the selected experiment configuration, outputs are written under the configured results directory.

For a reliable comparison between MAT-NAHT and POAM, report at least:

- Environment and map name
- Number and type of controlled/uncontrolled teammates
- Visibility-mask mode and its parameters
- Random seeds and number of independent runs
- Environment steps and wall-clock training time
- Mean evaluation return and variability across seeds
- Success/win rate where appropriate
- Checkpoint-selection and evaluation protocol
- Whether the visibility mask is fixed per episode or resampled during interaction

Do not claim that MAT-NAHT outperforms POAM unless the comparison has been run under a documented and fair evaluation protocol.

## Reproducibility and Current Limitations

- The uploaded archive does **not** include a README, trained-result figures, or GIF/video rollout assets. Add representative visualizations after generating them; do not use unverified figures as experimental evidence.
- The current default visibility generator samples a random gate with probability 0.7, keeps self-visibility, and uses no distance threshold or additional occlusion unless those arguments are explicitly passed.
- The `neighbor_mask` path can override the generated visibility mask when that key exists in the episode batch scheme.
- Visibility masks are generated in the agent forward path. For controlled ablations, expose visibility parameters in YAML and define deterministic mask sampling behavior.
- The project includes legacy environment dependencies. Exact Python/PyTorch/SMAC versions should be pinned after confirming a working installation.
- Some pretrained agent paths in the open-team configuration files are machine- or experiment-specific. Update them to valid local checkpoints before evaluation.
- The shell script `train_naht.sh` selects POAM, not MAT-NAHT, despite its filename.
- No experiment metrics were inferred from the source code. Add learning curves, confidence intervals, evaluation tables, and rollout videos only after running experiments.
- This repository includes upstream code under the MIT license. Preserve the upstream license and attribution when redistributing modified copies.

## Acknowledgements

This project builds on the NAHT/EPyMARL research codebase and its support for open-team multi-agent reinforcement learning.

Thanks to the authors of the following research works for making the underlying ideas and research resources available:

- Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone, **N-Agent Ad Hoc Teamwork**, NeurIPS 2024.
- Caroline Wang, Di Yang Shi, Elad Liebman, Ishan Durugkar, Arrasy Rahman, and Peter Stone, **Sequence Modeling for N-Agent Ad Hoc Teamwork**, arXiv:2506.05527.

## Citation

If you use the underlying research ideas, cite the corresponding papers. If this repository is a derivative of the original NAHT codebase, retain its required attribution and make your modifications clear.

**N-Agent Ad Hoc Teamwork (POAM):**

```bibtex
@inproceedings{wang2024naht,
  title     = {N-Agent Ad Hoc Teamwork},
  author    = {Wang, Caroline and Rahman, Arrasy and Durugkar, Ishan and Liebman, Elad and Stone, Peter},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

**Sequence Modeling for N-Agent Ad Hoc Teamwork (MAT-NAHT):**

```bibtex
@misc{wang2025sequencemodelingnagentad,
  title         = {Sequence Modeling for N-Agent Ad Hoc Teamwork},
  author        = {Wang, Caroline and Shi, Di Yang and Liebman, Elad and Durugkar, Ishan and Rahman, Arrasy and Stone, Peter},
  year          = {2025},
  eprint        = {2506.05527},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2506.05527}
}
```
