# Kanade
Kanade is a batch correction method explicitly designed to disentangle batch effects and the effects across conditions. Kanade can selectively correct the batch effects while preserving biologially meaningful conditional effects.

## Requirements / Notes

- GPU is recommended. The main tested environment is PyTorch with CUDA 12.4 (cu124).
- CPU-only usage is intended to work, but it is not fully validated yet (performance may be very slow and some configurations may fail).


## Installation

### 1. Install PyTorch

#### With GPU (Recommended)

`pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio`


#### CPU Only (Not recommended)

`pip install torch torchvision torchaudio`


### 2. Install Kanade

`pip install "git+https://github.com/ShuntaSakaguchi/kanade.git@main"`

## Quick Start
TBA


## Citation
If you use Kanade in your work, please cite this bioRxiv preprint.

Shunta Sakaguchi, Masato Tsutsumi, Kentaro Nishi Honda Naoki, Disentanglement of batch effects and biological signals across conditions in the single-cell transcriptome, bioRxiv, 2025 [link](https://www.biorxiv.org/content/10.1101/2025.04.10.648296v1)
