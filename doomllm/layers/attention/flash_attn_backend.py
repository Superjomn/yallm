from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from flash_attn import flash_attn_with_kvcache

from .attn_backend import AttentionBackend

if TYPE_CHECKING:
    from doomllm.memory.memory_pool import ReqToTokenPool
    from doomllm.scheduler.batch import Batch


@dataclass
class FlashAttentionMetadata:
    """metadata for flash_attn with kvcache"""

    cu_seqlens_q: Optional[torch.Tensor] = None
    max_seq_len_q: int = 0
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    window_size: tuple[int, int] = (-1, -1)
    page_table: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend"""

    def __init__(
        self,
        max_context_len: int,
        device: str,
        req_to_token_pool: ReqToTokenPool,
    ):
        super().__init__()

        self.metadata: Optional[FlashAttentionMetadata] = None
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token_pool = req_to_token_pool

    def init_metadata(self, batch: Batch):
        """initialize the metadata for the forward pass"""
        metadata = FlashAttentionMetadata()

        extend_seq_lens = batch.extend_prefix_lens
        seqlens = batch.seqlens
        metadata.cache_seqlens = seqlens.to(torch.int32)
        batch_size = len(seqlens)
        device = self.device
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cusum(seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.max_seq_len_k = seqlens.max().item()

        metadata.page_table = batch.req_to_token_pool.get_page_table(
            indices=batch.req_pool_indices,
            max_seq_len=self.max_context_len,
        )

        assert batch.forward_mode.is_decode() or batch.forward_mode.is_extend()

        if batch.forward_mode.is_decode():
            # all the query lengths are 1
            metadata.cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            )
        else:
            extend_no_prefix = not any(batch.extend_prefix_lens)
            if not extend_no_prefix:
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(torch.tensor(extend_seq_lens, dtype=torch.int32), dim=0), (1, 0)
                )
            else:
                # only prefilling
                metadata.cu_seqlens_q = metadata.cu_seqlens_k
            metadata.max_seq_len_q = seqlens.max().item()

        self.metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        batch: Batch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """forward pass for extend"""
        assert self.metadata is not None
        assert self.metadata.cu_seqlens_q is not None
        assert self.metadata.cu_seqlens_k is not None

        if self.metadata.max_seq_len_q < q.size(1):
            self.metadata.max_seq_len_q = q.size(1)

        out_cache_loc = batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.req_to_token_pool.write_kv_buffer(
                    layer_id=batch.layer_id,
                    loc=out_cache_loc,
                    cache_k=k,
                    cache_v=v,
                    k_scale=batch.k_scale,
                    v_scale=batch.v_scale,
                )

        metadata = self.metadata

        # TODO: calculate the window size
        window_size = (-1, -1)

        kv_cache = batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        out = flash_attn_with_kvcache(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache=key_cache.unsqueeze(1),
            v_cache=value_cache.unsqueeze(1),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seq_len_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            casual=True,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=layer.k_scale,
            v_descale=layer.v_scale,
        )

        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        batch: Batch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if k is not None and v is not None and save_kv_cache:
            out_cache_loc = batch.out_cache_loc
            batch.token_to_kv_pool.write_kv_buffer(
                layer_id=batch.layer_id,
                loc=out_cache_loc,
                cache_k=k,
                cache_v=v,
                k_scale=batch.k_scale,
                v_scale=batch.v_scale,
            )

        kv_cache = batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache, value_cache = kv_cache[0], kv_cache[1]

        metadata = self.metadata

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # TODO: calculate the window size
        window_size = (-1, -1)

        out = flash_attn_with_kvcache(
            q=q,
            k_cache=key_cache.unsqueeze(1),
            v_cache=value_cache.unsqueeze(1),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            casual=True,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=layer.k_scale,
            v_descale=layer.v_scale,
        )

        return out.view(-1, layer.tp_q_head_num * layer.head_dim)
