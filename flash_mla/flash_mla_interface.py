from typing import Optional, Tuple
import dataclasses

import torch

import flash_mla.cuda as flash_mla_cuda

@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (1), dtype torch.int32.


def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v. Must be 512
        sched_meta: FlashMLASchedMeta, return by get_mla_metadata. You may reuse the same sched_meta across different invocations, but only when the tensor shapes and the values of cache_seqlens remain the same.
        num_splits_placeholder: must be "None" (to be compatible with the old interface).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Whether to apply causal attention mask.
    
    For DeepSeek V3 and DeepSeek V3.1:
        head_dim should be 576 while head_dim_v should be 512.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"
    assert block_table is not None and cache_seqlens is not None, "block_table and cache_seqlens must be provided"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        # Initialize the tile scheduler metadata during the first invocation.
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],

            causal,
        )
    else:
        # Check whether the input arguments are consistent with sched_meta
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg

    out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
        q, k_cache, head_dim_v,
        cache_seqlens, block_table,
        softmax_scale, causal,
        sched_meta.tile_scheduler_metadata, sched_meta.num_splits
    )
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits
    return (out, lse)


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # Make lse contiguous on seqlen dim
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    max_seqlen_qo_aligned = (max_seqlen_qo + 7) // 8 * 8
    bs = cu_seqlens_qo.shape[0] - 1
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO and scaled_lse
    if num_qo_heads != num_kv_heads:
        workspace_bytes += 2 * kv_total_len * num_qo_heads * (head_dim_qk + head_dim_vo)  # dKV_acc
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )
