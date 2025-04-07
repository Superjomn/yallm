"""
The batch information.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from yallm.memory.memory_pool import ReqToTokenPool

# Referenced much idea from sglang's forward_batch_info.py


class BatchMode(IntEnum):
    """
    The mode of the batch.
    """

    # Prefill only
    PREFILL = auto()

    # Decode only
    DECODE = auto()

    # Mixing both EXTEND and DECODE.
    MIXED = auto()

    def is_prefill(self) -> bool:
        return self == BatchMode.PREFILL

    def is_decode(self) -> bool:
        return self == BatchMode.DECODE

    def is_mixed(self) -> bool:
        return self == BatchMode.MIXED


@dataclass
class Batch:
    """
    The batch information.
    """

    forward_mode: BatchMode

    batch_size: int

    input_ids: torch.Tensor

    # for decoding
    decode_seq_lens: Optional[torch.Tensor] = None

    # for extending
    extend_num_tokens: Optional[int] = None
    extend_prefix_lens: Optional[torch.Tensor] = None

    req_to_token_pool: Optional[ReqToTokenPool] = None

    # the indices of the requests in the req_to_token_pool
    req_pool_indices: Optional[torch.Tensor] = None
