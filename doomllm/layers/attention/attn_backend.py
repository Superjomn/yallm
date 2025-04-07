"""
Attention backend interface.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from doomllm.scheduler.batch import Batch


class AttentionBackend(ABC):
    """
    Interface for attention backend.
    """

    def init_forward_metadata(self, batch: Batch):
        raise NotImplementedError

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: Batch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, batch, save_kv_cache)
        elif batch.forward_mode.is_extend():
            return self.forward_extend(q, k, v, batch)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

    @abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: Batch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for decode.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Forward pass for extend.
        """
        raise NotImplementedError
