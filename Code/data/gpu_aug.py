"""GPU augmentation: random 224 crops, flips, and format equalization. Owner: Modeling.

Every batch arrives from the loader as raw pixels: uint8 [B, 256, 256, 3] (height, width, channels).
This module moves it to the GPU and turns it into the float tensor the network expects.

Training (`GpuAugment.train`):
  1. random 224x224 crop, a different offset for every image
  2. random horizontal flip
  3. format equalization: JPEG at a random quality in [65, 95], applied to real AND fake images
  4. ImageNet normalization (the backbones are ImageNet-pretrained)

Evaluation (`GpuAugment.eval`):
  center 224 crop, then optionally:
  - a fixed JPEG quality for every image         (E4 robustness sweep)
  - JPEG only on the images selected by a mask   (E3 format flip: compress fakes only)

Why step 3 matters: in OpenFake most reals are JPEG and most fakes are PNG (results/header_summary_core_train.csv).
Without it, a model can learn "JPEG artifacts => real" instead of anything about how the image was made.
Compressing both classes removes that cue. Cropping at a random offset before compressing also means the new
8x8 JPEG grid does not line up with any grid left in the original file.

Idea reference: format bias in fake-image datasets and JPEG equalization as a control,
Grommelt et al., "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets" (2024).
ImageNet mean/std: the standard values used for torchvision's ImageNet-pretrained models.

JPEG runs on the GPU through torchvision.io.encode_jpeg / decode_jpeg when the batch is on CUDA,
and on the CPU otherwise, so the same code is tested locally without a GPU.

    python -m data.gpu_aug          # quick self-check on random images
"""
from __future__ import annotations

import torch
from torchvision.io import decode_jpeg, encode_jpeg

CROP = 224
JPEG_Q_RANGE = (65, 95)                 # inclusive range for training
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_device(batch: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """uint8 [B, H, W, 3] from the loader -> uint8 [B, 3, H, W] on `device` (channels first, as PyTorch expects)."""
    return batch.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()


def random_crop(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Crop a size x size patch from each image at its own random offset. x: [B, C, H, W]."""
    b, _, h, w = x.shape
    tops = torch.randint(0, h - size + 1, (b,), generator=gen).tolist()
    lefts = torch.randint(0, w - size + 1, (b,), generator=gen).tolist()
    return torch.stack([x[i, :, t:t + size, l:l + size] for i, (t, l) in enumerate(zip(tops, lefts))])


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Crop the central size x size patch of every image. x: [B, C, H, W]."""
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def random_hflip(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Mirror each image left-right with probability 0.5."""
    flip = (torch.rand(x.shape[0], generator=gen) < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def jpeg(x: torch.Tensor, quality: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """JPEG-compress and decode images, each at its own quality.

    x:       uint8 [B, 3, H, W]
    quality: int [B], JPEG quality per image (1-100)
    mask:    bool [B] or None; only images where mask is True are compressed, the rest are returned unchanged

    encode_jpeg takes one quality per call, so images are grouped by quality: at most 31 calls per
    training batch (qualities 65..95), each one a batched call.
    """
    out = x.clone()
    quality = quality.cpu()
    selected = torch.ones(x.shape[0], dtype=torch.bool) if mask is None else mask.cpu().bool()
    for q in torch.unique(quality[selected]).tolist():
        idx = torch.nonzero(selected & (quality == q)).flatten().tolist()
        encoded = encode_jpeg([x[i] for i in idx], quality=int(q))
        encoded = [e.cpu() for e in encoded]          # decode_jpeg reads the compressed bytes from CPU memory
        decoded = decode_jpeg(encoded, device=x.device)
        for i, img in zip(idx, decoded):
            out[i] = img
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 3, H, W] -> float32 with ImageNet mean and std, the input scale of a pretrained backbone."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255.0 - mean) / std


class GpuAugment:
    """Turns loader batches into network input, for training or evaluation.

    aug = GpuAugment(device="cuda", seed=0)
    x = aug.train(images)                              # training batch
    x = aug.eval(images)                               # clean evaluation
    x = aug.eval(images, jpeg_quality=50)              # E4: every image at Q=50
    x = aug.eval(images, jpeg_quality=75, jpeg_mask=is_fake)   # E3 format flip: fakes only
    """

    def __init__(self, device: torch.device | str = "cuda", crop: int = CROP,
                 jpeg_range: tuple[int, int] | None = JPEG_Q_RANGE, hflip: bool = True,
                 seed: int = 0, channels_last: bool = True):
        self.device = torch.device(device)
        self.crop = crop
        self.jpeg_range = jpeg_range       # None turns format equalization off (protocols N and plain C in E3)
        self.hflip = hflip
        self.channels_last = channels_last
        self.gen = torch.Generator().manual_seed(seed)   # random choices are drawn on the CPU, so runs repeat exactly

    def _finish(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize(x)
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    def train(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 [B, H, W, 3] from the loader -> augmented float [B, 3, crop, crop] on the device."""
        x = to_device(images, self.device)
        x = random_crop(x, self.crop, self.gen)
        if self.hflip:
            x = random_hflip(x, self.gen)
        if self.jpeg_range is not None:
            lo, hi = self.jpeg_range
            q = torch.randint(lo, hi + 1, (x.shape[0],), generator=self.gen)
            x = jpeg(x, q)
        return self._finish(x)

    def eval(self, images: torch.Tensor, jpeg_quality: int | None = None,
             jpeg_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic evaluation input: center crop, then an optional fixed-quality JPEG."""
        x = center_crop(to_device(images, self.device), self.crop)
        if jpeg_quality is not None:
            q = torch.full((x.shape[0],), int(jpeg_quality))
            x = jpeg(x.contiguous(), q, jpeg_mask)
        return self._finish(x)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = torch.randint(0, 256, (8, 256, 256, 3), dtype=torch.uint8)
    aug = GpuAugment(device=device, seed=0)
    x = aug.train(images)
    print(f"device={device} train batch: {tuple(x.shape)} {x.dtype}, mean={x.mean():.3f}")
    print(f"eval batch: {tuple(aug.eval(images).shape)}, eval with JPEG Q=50: {tuple(aug.eval(images, 50).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
