"""
The batch information.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Optional

import torch

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

    batch_size: int

    input_ids: torch.Tensor

    # for decoding
    decode_seq_lens: Optional[torch.Tensor] = None

    # for extending
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
