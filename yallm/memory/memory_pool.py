import abc
from typing import Optional

import numpy as np
import torch

# Two levels of memory management for KV Cache:
# 1. ReqToTokenPool: the mapping from request to its token locations.
# 2. TokenToKVPool: the mapping from token locations to memory locations.


class ReqToTokenPool:
    """
    A memory pool that maps requests to their token locations.

    Parameters:
        size: the size of the memory pool.
        max_seq_len: the maximum sequence length.
        device: the device to store the memory pool.
    """

    def __init__(self, size: int, max_seq_len: int, device: str):
        self.size = size
        self.max_seq_len = max_seq_len
        self.device = device

        self._free_slots = list(range(size))
        self._map = torch.zeros((size, max_seq_len), dtype=torch.int32, device=device)

    def update(
        self,
        indices: list[int],
        values: list[int],
    ):
        self._map[indices] = values

    @property
    def free_size(self) -> int:
        """
        The number of free slots in the memory pool.
        """
        return len(self._free_slots)

    def alloc(self, size: int) -> Optional[list[int]]:
        """
        Allocate a list of indices from the free slots.
        """
        if size > self.free_size:
            return None

        indices = self._free_slots[:size]
        self._free_slots = self._free_slots[size:]
        return indices

    def free(self, indices: list[int] | int):
        """
        Free a list of indices from the memory pool.
        """
        if isinstance(indices, int):
            indices = [indices]

        self._free_slots.extend(indices)

    def clear(self):
        """
        Clear the memory pool.
        """
        self._free_slots = list(range(self.size))


class KVCacheInterface(abc.ABC):
    """
    Interface for KV cache.
    """

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> torch.Tensor:
        pass


class TokenToKVPoolAllocator:
    """
    A memory pool that maps token locations to KV cache locations.
    """

    def __init__(self, size: int, max_seq_len: int, device: str, kvcache: KVCacheInterface):
        self.size = size
        self.max_seq_len = max_seq_len
        self.device = device

        # Used tensor for more efficient indices operations than list.
        self._free_slots: Optional[torch.Tensor] = None

        # Whether the current free operation is in a batch.
        self._in_free_batch = False

        # Batch of free indices.
        self._free_batch: list[torch.Tensor] = []

        self._kvcache = kvcache

    @property
    def free_size(self) -> int:
        return len(self._free_slots)

    def alloc(self, size: int):
        if size > self.free_size:
            return None

        indices = self._free_slots[:size]
        self._free_slots = self._free_slots[size:]
        return indices

    def free(self, indices: torch.Tensor):
        """
        Free a list of indices from the memory pool.
        """
        if indices.numel() == 0:
            return

        if self._in_free_batch:
            self._free_slots = torch.cat([self._free_slots, indices])
        else:
            self._free_batch.append(indices)

    def free_batching_begin(self):
        """
        Begin a batch of free operations.
        """
        self._in_free_batch = True
        self.free_group = []

    def free_batching_end(self):
        """
        End a batch of free operations.
        """
        self._in_free_batch = False
        if self._free_batch:
            self._free_slots = torch.cat(self._free_batch)

    def clear(self):
        """
        Clear the memory pool.
        """
        self._free_slots = torch.arange(
            1,
            self.size + 1,
            dtype=torch.int64,
            device=self.device,
        )
        self._in_free_batch = False
        self._free_batch = []


class MHAToenToKvPool(KVCacheInterface):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        self._key_buffers = [
            torch.zeros(
                (self.size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        # [size, head_num, head_dim] for each layer
        self._value_buffers = [
            torch.zeros(
                (self.size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def get_kv_size_bytes(self) -> tuple[int, int]:
        """
        Get the size of the KV cache in bytes.
        """
        return (
            self.size * self.head_num * self.head_dim * self.dtype.size,
            self.size * self.head_num * self.head_dim * self.dtype.size,
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """
        Get the key buffer for a given layer.
        """
        return self._key_buffers[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """
        Get the value buffer for a given layer.
        """
        return self._value_buffers[layer_id]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
