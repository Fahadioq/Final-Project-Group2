"""The proposal's two heads: one real/fake logit (BCE) and a softmax over known
generators + unknown-generator + known real sources + unknown-real (CE, weight 0.5); binary-only variant for the ablation.

Owner: Modeling. Not implemented yet -- see the interfaces in Code/common/schemas.py.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("heads.py is a placeholder for the Modeling role")


if __name__ == "__main__":
    raise SystemExit(main())
