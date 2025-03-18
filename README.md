# Debiasing Diffusion Model (DDM)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
## Introduction
Debiasing Diffusion Model (DDM) is a framework designed to mitigate bias in latent diffusion models (LDMs).
1. Fair Image Generation
    Reduces demographic disparities in text-to-image diffusion models.
2. Flexible Debiasing:
    Works for both scenarios:  
    1. When sensitive attributes are unknown, it balances the distribution of generated images across demographic categories.
    2. When sensitive attributes are explicitly given as conditions, it mitigates performance disparities across different demographic groups while preserving the intended control over generation.
3. Efficient Debiasing
    Supports LoRA-based fine-tuning, allowing for lightweight and efficient adaptation to new datasets without full model retraining.
## Repository Structure  
```
DebiasingDiffusionModel/
│── examples/               # Example scripts
│── src/                    # Main source code directory
|   │── ddm                 # Core DDM package
|   |   │── models/         # Model architectures and implementations 
|   |   └── __init__.py
│── utils/                  # Utility scripts and helper functions
│── README.md
│── requirements.txt
└── LICENSE
```
## Installation
### Cloning the Repository
```shell
$ git clone https://github.com/chun61205/DebiasingDiffusionModel.git
$ cd DebiasingDiffusionModel
```
### Dependencies
```shell
$ pip install -r requirements.txt
```
## Usage
### Training the Model
```shell
$ bash examples/trainDreambooth.sh
```
### Inferencing
```shell
$ bash examples/pipeline.sh
```
## Citation
If you find this repository useful for your research, please cite our work:
```bibtex
@misc{huang2025debiasingdiffusionmodelenhancing,
      title={Debiasing Diffusion Model: Enhancing Fairness through Latent Representation Learning in Stable Diffusion Model}, 
      author={Lin-Chun Huang and Ching Chieh Tsao and Fang-Yi Su and Jung-Hsien Chiang},
      year={2025},
      eprint={2503.12536},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.12536}, 
}
```