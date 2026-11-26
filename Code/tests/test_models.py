"""Self-contained checks for the two-head detector and its loss (no pytest, no GPU, no weight download).

    cd Code && python -m tests.test_models
"""
from __future__ import annotations

import torch

from models.backbones import BACKBONES, build_backbone, input_stats
from models.heads import NO_CLASS, Detector, build_detector, detector_loss, param_groups, probabilities

torch.manual_seed(0)
X = torch.randn(4, 3, 224, 224)


def test_backbones_give_feature_vectors():
    for name in BACKBONES:
        net, n = build_backbone(name, pretrained=False)
        assert net(X[:2]).shape == (2, n), name
        assert input_stats(net) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), name


def test_two_head_output_shapes():
    out = build_detector("resnet50", n_classes=7, pretrained=False)(X)
    assert out["logit"].shape == (4,) and out["attr_logits"].shape == (4, 7)
    probs = probabilities(out)
    assert torch.allclose(probs["attr_probs"].sum(1), torch.ones(4), atol=1e-5)
    assert ((probs["p_fake"] > 0) & (probs["p_fake"] < 1)).all()


def test_binary_only_variant():
    out = build_detector("resnet50", n_classes=0, pretrained=False)(X)
    assert set(out) == {"logit"} and "attr_probs" not in probabilities(out)


def _toy_out(k: int = 3):
    return {"logit": torch.tensor([2.0, -1.0, 0.5, 3.0], requires_grad=True),
            "attr_logits": torch.randn(4, k, requires_grad=True)}


def test_bce_mask_excludes_images():
    out = _toy_out()
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    attr = torch.tensor([0, 1, 2, 1])
    mask = torch.tensor([True, True, True, False])
    total, _ = detector_loss(out, label, attr, mask)
    total.backward()
    assert out["logit"].grad[3] == 0                    # masked image: no gradient into the binary head
    assert out["attr_logits"].grad[3].abs().sum() > 0   # but it still trains the attribution head


def test_no_class_is_ignored():
    out = _toy_out()
    label = torch.zeros(4)
    attr = torch.tensor([0, NO_CLASS, 2, NO_CLASS])
    total, _ = detector_loss(out, label, attr, torch.ones(4, dtype=torch.bool))
    total.backward()
    assert out["attr_logits"].grad[1].abs().sum() == 0 and out["attr_logits"].grad[3].abs().sum() == 0
    _, parts = detector_loss(_toy_out(), label, torch.full((4,), NO_CLASS), torch.zeros(4, dtype=torch.bool))
    assert parts["loss"] == 0.0                          # nothing to learn from: zero loss, no NaN


def test_param_groups_cover_every_parameter_once():
    model = build_detector("resnet50", n_classes=5, pretrained=False)
    groups = param_groups(model, lr_backbone=1e-4, lr_heads=1e-3)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3 and len(groups[1]["params"]) == 4


def test_model_can_learn_a_toy_task():
    """A few steps on a fixed batch must lower the loss: gradients reach both heads and the backbone."""
    model = Detector(*build_backbone("resnet18", pretrained=False), n_classes=3)
    opt = torch.optim.AdamW(param_groups(model, 1e-3, 1e-2))
    x = torch.randn(8, 3, 64, 64)
    label = torch.tensor([0, 1] * 4, dtype=torch.float32)
    attr = torch.tensor([0, 1, 2, 1, 0, 2, 0, 1])
    mask = torch.ones(8, dtype=torch.bool)
    first = None
    for _ in range(15):
        loss, parts = detector_loss(model(x), label, attr, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else parts["loss"]
    assert parts["loss"] < 0.5 * first, (first, parts["loss"])


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
