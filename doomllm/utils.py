from typing import Optional

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    attrs: Optional[dict] = None,
) -> torch.Tensor:
    if attrs is None:
        return
    for key, value in attrs.items():
        if hasattr(weight, key):
            raise ValueError(f"Weight already has attribute {key}")
        setattr(weight, key, value)
    return weight


def add_prefix(prefix: str, name: str) -> str:
    if prefix:
        return f"{prefix}.{name}"
    return name
