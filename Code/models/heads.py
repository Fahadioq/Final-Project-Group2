"""The proposal's two-head detector and its loss. Owner: Modeling.

One shared backbone, two small linear heads on its feature vector:

  binary head       1 logit  -> sigmoid -> p(fake)                     trained with BCE
  attribution head  K logits -> softmax -> which source made the image  trained with cross-entropy
                    K = known generators + "unknown-generator" + known real sources + "unknown-real"
                    (the class list comes from the split file, see data/memmap_dataset.class_list)

Loss = BCE + attr_weight * CE, with attr_weight = 0.5 (PLAN section 7, E5).

Two masks keep the attribution head from changing the detection experiments:
  - bce_mask = False: auxiliary real photos used only to teach "unknown-real". They are left out of the BCE
    term, so every run's binary detector sees the same real pool.
  - attr class -1: the image has no attribution target. It is left out of the CE term.

n_classes = 0 builds the binary-only model used in the E5 ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import build_backbone

ATTR_WEIGHT = 0.5
NO_CLASS = -1


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, n_features: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(n_features, 1)
        self.attr_head = nn.Linear(n_features, n_classes) if n_classes > 0 else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))            # [B, n_features]
        out = {"logit": self.binary_head(features).squeeze(1)}   # [B]
        if self.attr_head is not None:
            out["attr_logits"] = self.attr_head(features)    # [B, K]
        return out


def build_detector(backbone: str = "resnet50", n_classes: int = 0, pretrained: bool = True,
                   dropout: float = 0.0) -> Detector:
    net, n_features = build_backbone(backbone, pretrained)
    return Detector(net, n_features, n_classes, dropout)


def detector_loss(out: dict[str, torch.Tensor], label: torch.Tensor, attr: torch.Tensor,
                  bce_mask: torch.Tensor, attr_weight: float = ATTR_WEIGHT) -> tuple[torch.Tensor, dict]:
    """Total loss and its parts (for logging).

    label:    float [B], 1 = fake
    attr:     int [B], attribution class index, NO_CLASS (-1) for no target
    bce_mask: bool [B], False for images that must not train the binary head
    """
    per_image = F.binary_cross_entropy_with_logits(out["logit"].float(), label.float(), reduction="none")
    keep = bce_mask.float()
    bce = (per_image * keep).sum() / keep.sum().clamp(min=1.0)   # mean over the kept images only

    ce = torch.zeros((), device=per_image.device)
    if "attr_logits" in out and (attr != NO_CLASS).any():
        ce = F.cross_entropy(out["attr_logits"].float(), attr, ignore_index=NO_CLASS)

    total = bce + attr_weight * ce
    return total, {"loss": total.item(), "bce": bce.item(), "ce": ce.item()}


def probabilities(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Logits -> p_fake [B] and, for the two-head model, attribution probabilities [B, K]."""
    probs = {"p_fake": torch.sigmoid(out["logit"].float())}
    if "attr_logits" in out:
        probs["attr_probs"] = torch.softmax(out["attr_logits"].float(), dim=1)
    return probs


def param_groups(model: Detector, lr_backbone: float = 1e-4, lr_heads: float = 1e-3,
                 weight_decay: float = 0.05) -> list[dict]:
    """Two learning rates for AdamW (PLAN section 8): small for the pretrained backbone, so its ImageNet
    features change slowly, and 10x larger for the new heads, which start from random weights."""
    heads = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    return [
        {"params": list(model.backbone.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": heads, "lr": lr_heads, "weight_decay": weight_decay},
    ]
