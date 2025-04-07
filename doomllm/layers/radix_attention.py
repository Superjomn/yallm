# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Optional

from torch import nn

from doomllm.scheduler.batch import Batch


class RadixAttention(nn.Module):
    """Radix attention"""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.v_head_dim = v_head_dim
        self.sliding_window_size = sliding_window_size

        self.k_scale: Optional[float] = None
        self.v_scale: Optional[float] = None

    def forward(self, q, k, v, batch: Batch, save_kv_cache: bool = True):
        if k is not None:
            assert v is not None
            k = k.view(-1, self.tp_q_head_num, self.head_dim)
            v = v.view(-1, self.tp_q_head_num, self.head_dim)
        return batch.attn_backend.forward(
            q=q,
            k=k,
            v=v,
            layer=self,
            batch=batch,
            save_kv_cache=save_kv_cache,
        )
