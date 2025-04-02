# Test major features of flashattn for inference mode

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def test_flash_attn_varlen_func_without_kvcache(
    device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16
):
    """Test variable length FlashAttention implementation.

    Args:
        device: Device to run the test on
        seed: Random seed for reproducibility
        batch_size: Number of sequences in batch
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head


    The flash_attn_varlen_func is for prefilling phase.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate total hidden dimension
    hidden_dim = num_heads * head_dim

    # Generate random sequence lengths between 10 and 100
    seq_len = torch.randint(10, 100, (batch_size, 1), device=device)
    max_seq_len = torch.max(seq_len).item()
    total_seq_len = torch.sum(seq_len).item()

    # All of the q,k,v packs all the sequence into one tensor
    # Create query, key, value tensors (total_seq_len, num_heads, head_dim)
    q = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    # Remove the extra dimension from seq_len
    seq_len = seq_len.squeeze(1)

    # Create cumulative sequence lengths with leading 0
    # This creates offsets: [0, len1, len1+len2, len1+len2+len3, ...]
    cu_seqlens_q = torch.cumsum(seq_len, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens_q])
    cu_seqlens_k = cu_seqlens_q.clone()  # Keys have same lengths as queries

    # Run flash attention with variable length sequences
    res = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        dropout_p=0.0,
        return_attn_probs=True,
    )

    output = res[0]
    attn_probs = res[1]
    S_mask = res[2]

    # Basic validation - check output shape matches input shape
    assert (
        output.shape == q.shape
    ), f"Output shape {output.shape} doesn't match input shape {q.shape}"

    # Verify output is not all zeros or NaNs
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.any(output != 0), "Output is all zeros"

    print("output", output)
    print("attn_probs", attn_probs)
    print("S_mask", S_mask)

    return output


def test_flash_attn_varlen_func_with_kvcache(
    device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16
):
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate total hidden dimension
    hidden_dim = num_heads * head_dim

    # Generate random sequence lengths between 10 and 100
    seq_lens = torch.randint(10, 100, (batch_size, 1), device=device)
    max_seq_len = torch.max(seq_lens).item()
    total_seq_len = torch.sum(seq_lens).item()
    paged_kv_block_size = 256

    max_k_seq_len = 100
    k_seq_lens = torch.randint(0, max_k_seq_len, (batch_size, 1), device=device)
    total_k_seq_len = torch.sum(k_seq_lens).item()

    # All of the q,k,v packs all the sequence into one tensor
    # Create query, key, value tensors (total_seq_len, num_heads, head_dim)
    q = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    k_cache_paged, v_cache_paged, block_table = generate_block_kvcache(
        max_k_seq_len + 100,  # room for new tokens
        paged_kv_block_size,
        batch_size,
        num_heads,
        head_dim,
        device,
        dtype=torch.float16,
    )

    # Remove the extra dimension from seq_len
    seq_lens = seq_lens.squeeze(1)
    k_seq_lens = k_seq_lens.squeeze(1)
    # Create cumulative sequence lengths with leading 0
    # This creates offsets: [0, len1, len1+len2, len1+len2+len3, ...]
    cu_seqlens_q = create_culens(seq_lens, device)
    cu_seqlens_k = create_culens(k_seq_lens, device)

    # Run flash attention with variable length sequences
    res = flash_attn_varlen_func(
        q,
        k=k_cache_paged,
        v=v_cache_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_k_seq_len,
        block_table=block_table,
        dropout_p=0.0,
        return_attn_probs=True,
    )

    output = res[0]
    attn_probs = res[1]
    S_mask = res[2]

    print("output", output)
    print("attn_probs", attn_probs)
    print("S_mask", S_mask)


def create_culens(seq_lens: torch.Tensor, device: torch.device):
    cu_seqlens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens])
    return cu_seqlens


def test_flash_attn_with_kvcache(device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16):
    # The flash_attn_with_kvcache is for incremental decoding

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate total hidden dimension
    hidden_dim = num_heads * head_dim

    batch_size = 10

    # all the queries have the same length of 1
    q = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)

    # Generate random sequence lengths between 10 and 100
    max_seq_len_k = 100
    seq_lens_k = torch.randint(10, max_seq_len_k, (batch_size,), device=device, dtype=torch.int32)
    max_seq_len_k = torch.max(seq_lens_k).item()
    total_seq_len_k = torch.sum(seq_lens_k).item()

    paged_kv_block_size = 256
    k_cache_paged, v_cache_paged, block_table = generate_block_kvcache(
        max_seq_len_k,
        paged_kv_block_size,
        batch_size,
        num_heads,
        head_dim,
        device,
        dtype=torch.float16,
    )
    # quote from the doc of flash_attn_with_kvcache:
    # If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    # k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    # the previous step, and update them with the new keys/values from the current step, and do
    # attention with the updated cache, all in 1 kernel.

    # so there are two approaches to mange the new k,v:
    # 1. pass in the new k,v to the function, and it will update the kv cache inplace later
    #    - This requires the kv cache pages has enough space to store the new k,v
    # 2. update the kv cache inplace with the new k,v first, then call the function

    # Let's just try the 1st approach for simplicity

    # for the new k,v, their lengths are the same as the query, that is 1
    k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)

    res = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        k=k,
        v=v,
        cache_seqlens=seq_lens_k,
        block_table=block_table,
    )
    print("res", res)


def generate_block_kvcache(
    max_seqlen_k: int,
    paged_kv_block_size: int,
    max_batch_size: int,
    nheads_k: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Generate a block of KV cache for a given sequence length.
    """
    num_blocks = math.ceil(max_seqlen_k / paged_kv_block_size) * max_batch_size
    # the paged cache storage for kv cache
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    # a mapping table that maps the logical sequence positions and the physical memory blocks
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=max_batch_size,
    )
    return k_cache_paged, v_cache_paged, block_table


if __name__ == "__main__":
    # output = test_flash_attn_varlen_func()
    # print(f"Test passed! Output tensor shape: {output.shape}")
    # test_flash_attn_with_kvcache()
    # test_flash_attn_varlen_func_with_kvcache()
    test_flash_attn_with_kvcache()
