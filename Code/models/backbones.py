"""Pretrained backbones from timm. Owner: Modeling.

A backbone turns an image into one feature vector; the heads (models/heads.py) turn that vector into answers.
Each backbone is created with num_classes=0, which removes its ImageNet classifier and keeps the global
average pooling, so the output is a [B, n_features] tensor.

  resnet50         primary (PLAN decision 1): the classic CNN baseline in fake-image detection
  convnext_tiny    secondary: a modern CNN, already cached on the group instance
  efficientnet_b4  optional

All three were pretrained with ImageNet mean/std, which is the normalization in data/gpu_aug.py.
`input_stats` reads the values from timm, so train.py can refuse a backbone that expects something else.

Idea reference: ResNet-50 as the standard detector backbone, Wang et al., "CNN-generated images are
surprisingly easy to spot... for now" (CVPR 2020).
"""
from __future__ import annotations

import timm
import torch.nn as nn

BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k_384",
    "efficientnet_b4": "tf_efficientnet_b4.ns_jft_in1k",
}


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (backbone, n_features). `name` is a key of BACKBONES or any full timm model name."""
    timm_name = BACKBONES.get(name, name)
    backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    return backbone, backbone.num_features


def input_stats(backbone: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) the backbone was pretrained with, from timm's pretrained config."""
    cfg = backbone.pretrained_cfg
    return tuple(cfg["mean"]), tuple(cfg["std"])
