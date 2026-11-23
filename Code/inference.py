"""Inference API for the app: load(run_dir) -> Detector; Detector.predict(pil_image) -> dict.

Must preprocess exactly like evaluation: etl.views.view_c, then a center 224 crop.

Owner: Modeling. Not implemented yet -- see the interfaces in Code/common/schemas.py.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("inference.py is a placeholder for the Modeling role")


if __name__ == "__main__":
    raise SystemExit(main())
