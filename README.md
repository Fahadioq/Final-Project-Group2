# Cross-Generator Generalization in AI-Generated Image Detection

**DATS 6303 -- Deep Learning and Fundamentals of AI | Group 2**

## Overview
Generative models such as Stable Diffusion and Midjourney now produce photorealistic images that can be misused for misinformation. Neural networks can learn to separate real from generated images, but detectors trained on one generator often fail on generators unseen during training. Because new generators appear continually, this weakness limits practical use. Our goal is to measure the effectiveness of CNNs trained on one subset of generators against those they have yet to see.

## Approach
- **Dataset:** [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (Livernoche et al., 2025) v.2.0
- **Model:** Various pretrained backbones
- **Framework:** PyTorch
- **Experiments:** in-distribution baseline, cross-generator evaluation, robustness to JPEG compression and resizing, and a shortcut-bias control.
- **Metrics:** F1-score,  the generalization gap

## Repository structure
- `Group_Proposal/` -- project proposal
- `Final-Group-Project-Report/` -- final group report (PDF)
- `Final-Group-Presentation/` -- presentation (PDF)
- `Code/` -- all code; [Code/README.md](Code/README.md) gives the run order and who owns what
- `results/` -- small CSV/JSON result tables (no images or checkpoints)
