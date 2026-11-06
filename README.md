# Cross-Generator Generalization in AI-Generated Image Detection

**DATS 6303 – Deep Learning and Fundamentals of AI · Group 2**

## Overview
Image generators such as Stable Diffusion and Midjourney now produce photorealistic images, and detectors trained on one generator often fail on generators they have never seen. This project fine-tunes a ResNet50 detector on images from one generator and measures how well it identifies images from unseen generators.

**Research question:** Can a detector trained on Stable Diffusion v1.4 images correctly identify images produced by other generators, and what does it actually learn?

## Approach
- **Dataset:** [GenImage](https://github.com/weili-thu/genimage) (Zhu et al., NeurIPS 2023): real ImageNet images and images from eight generators.
- **Model:** ImageNet-pretrained ResNet50, fully fine-tuned with a single-logit head (binary cross-entropy).
- **Framework:** PyTorch and torchvision, trained on GPU.
- **Experiments:** in-distribution baseline, cross-generator evaluation, robustness to JPEG compression and resizing, and a shortcut-bias control.
- **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC, and the generalization gap (drop in ROC-AUC on unseen generators).
