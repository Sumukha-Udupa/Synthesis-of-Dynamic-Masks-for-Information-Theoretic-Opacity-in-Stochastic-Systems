# Synthesis of Dynamic Masks for Information Theoretic Opacity in Stochastic Systems
[![DOI](https://zenodo.org/badge/871313574.svg)](https://doi.org/10.5281/zenodo.14835001)


## Description

In this work, we investigate the synthesis of dynamic information release mechanisms, referred to as 'masks', to reduce the information leakage from a stochastic system to an external observer.

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

## Usage

The setup for the Illustrative example, and Stochastic Gridworld Case Study as given in the paper can be run by running the appropriate script.

```bash
# To run Illustrative example
Examples/script_for_illustrative_example_results.py
# To run Stochastic Gridworld for No Masking and Final state Masking
Examples/script_for_pharma_example_no_masking_and_final_state_masking.py
# To run Stochastic Gridworld for Dynamic Mask synthesis
Examples/script_for_pharma_example_with_dynamic_masking.py
```

## Adaptation

- The code may be adapted for any other

- If any code modification needs to be done for their respective work, the Gridworld environment may be modified in the corresponding scripts:

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

If you found this repo useful, please cite our paper.
