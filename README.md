# Synthesis of Dynamic Masks for Information Theoretic Opacity in Stochastic Systems
[![DOI](https://zenodo.org/badge/871313574.svg)](https://doi.org/10.5281/zenodo.14835001)


## Description

In this work, we investigate the synthesis of dynamic information release mechanisms, called 'masks', to reduce the information leakage from a stochastic system to an external observer. Specifically, for a stochastic system, an observer aims to infer whether the final state of the system trajectory belongs to a set of secret states. The dynamic mask seeks to regulate sensor information to maximize the observer's uncertainty about the final state, a property known as final-state opacity. While existing supervisory control literature on dynamic masks primarily addresses qualitative opacity, we propose quantifying opacity in stochastic systems by conditional entropy, which is a measure of information leakage in information security. We then formulate a constrained optimization problem to synthesize a dynamic mask that maximizes final-state opacity under a total cost constraint on masking. To solve this constrained optimal dynamic mask synthesis problem, we develop a novel primal-dual policy gradient method. Additionally, we present a technique for computing the gradient of conditional entropy with respect to the masking policy parameters, leveraging observable operators in hidden Markov models. To demonstrate the effectiveness of our approach, we apply our method to an illustrative example and a stochastic grid world scenario, showing how our algorithm optimally enforces final-state opacity under cost constraints. 

## Installation

Clone the repository and install dependencies using the following commands. We also suggest you install pygame and PyTorch, and the necessary CUDA toolkit for using GPU. The code can be run on CPU alone with PyTorch.

```bash
# Clone the repository
https://github.com/Sumukha-Udupa/Synthesis-of-Dynamic-Masks-for-Information-Theoretic-Opacity-in-Stochastic-Systems.git

# Install dependencies
pip install -r requirements.txt

# Install PyTorch for CPU only.
pip install torch torchvision torchaudio

# Install PyTorch for CUDA-enabled GPUs (recommended for NVIDIA 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyGame
pip install pygame
```

Or alternatively use the Docker image:
[Docker Hub Repository](https://hub.docker.com/r/sumukhaudupa/iccps)

## Docker Image

The Docker image for this project is available on Docker Hub. You can pull it using:

```bash
docker pull sumukhaudupa/iccps
```
## Project Structure
```plaintext
|------ Examples
|        |--- experiment_plots
|        |--- logs_for_example
|        |--- pharmaceutical_example_multi_init_states_no_masking
|        |--- pharmaceutical_example_multi_init_states_final_state_masking
|        |--- pharmaceutical_example_multi_init_states_modified_obs
|        |--- running_example
|        |--- running_example_no_masking
|        |--- running_example_with_augmented_obs
|        |--- script_for_illustrative_example_results
|        |--- script_for_pharma_example_no_masking_and_final_state_masking
|        |--- script_for_pharma_example_with_dynamic_masking
|        
|
|------ setup_and_solvers
|        |--- gridworld_env_multi_init_states
|        |--- hidden_markov_model_of_P2_changed_observations
|        |--- markov_decision_process
|        |--- test_gradient_calculation_with_final_masking_policy
|        |--- test_gradient_entropy_calculations_modified_obs
|
```


## Usage

The setup for the Illustrative example and Stochastic Gridworld Case Study as given in the paper can be run by running the appropriate script.

```bash
# To run Illustrative example
Examples/script_for_illustrative_example_results.py
# To run Stochastic Gridworld for No Masking and Final state Masking
Examples/script_for_pharma_example_no_masking_and_final_state_masking.py
# To run Stochastic Gridworld for Dynamic Mask synthesis
Examples/script_for_pharma_example_with_dynamic_masking.py
```

## Adaptation

- If any code modification needs to be done for any other work, the Gridworld environment may be modified in the corresponding scripts:

  - `Examples/pharmaceutical_example_multi_init_states_no_masking.py`
  - `Examples/pharmaceutical_example_with_multiple_init_states_final_state_masking.py`
  - `Examples/pharmaceutical_example_with_multiple_init_states_modified_obs.py`

- The results of each run are stored in the folder `Examples/experiment_plots`, where each plot is named by the experiment numbers as given in the scripts.

## System Configuration

The following are the minimum system requirements (as the experiments were conducted on a system with the following configuration):

1. **Hardware Requirements**:

   - Intel Core i7 CPU @ 3.2GHz
   - 32 GB RAM
   - 8 GB RTX 3060 GPU

2. **Software Requirements**:

   - Python 3.9.7
   - PyTorch

## Citation

If you use this code in your research or found this useful, please cite the following paper:

**Synthesis of Dynamic Masks for Information-Theoretic Opacity in Stochastic Systems**  
Author(s): Sumukha Udupa, Chongyang Shi, Jie Fu  
Published in: arXiv preprint arXiv:2502.10552, 2025  
[Link to Paper](https://arxiv.org/abs/2502.10552)

```bibtex
@article{udupa2025synthesis,
  title={Synthesis of Dynamic Masks for Information-Theoretic Opacity in Stochastic Systems},
  author={Udupa, Sumukha and Shi, Chongyang and Fu, Jie},
  journal={arXiv preprint arXiv:2502.10552},
  year={2025}
}
```

