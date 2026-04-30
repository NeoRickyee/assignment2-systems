from typing import Any

import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        q: Float32[Tensor, "... N_QUERIES D"],
        k: Float32[Tensor, "... N_KEYS D"],
        v: Float32[Tensor, "... N_KEYS D"],
        is_causal=False,
    ):
        q_size = q.size()
        k_size = k.size()
        ctx.is_causal = is_causal
        
        D = q_size[-1]
        N_QUERIES = q_size[-2]
        N_KEYS = k_size[-2]
        assert D == k_size[-1], "D must match"
        
        batch_shape = q.shape[:-2]
        
        scale = 1.0 / (D ** 0.5)
        
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        stride_qb = D * N_QUERIES
        stride_kb = D * N_KEYS
        stride_vb = D * N_KEYS
        stride_ob = D * N_QUERIES
        stride_lb = N_QUERIES
        
        stride_qq = D
        stride_kk = D
        stride_vk = D
        stride_oq = D
        stride_lq = 1

        stride_qd = 1
        stride_kd = 1
        stride_vd = 1
        stride_od = 1

        o: Float32[Tensor, "... N_QUERIES D"] = torch.empty_like(q)
        l: Float32[Tensor, "... N_QUERIES"] = torch.empty(
            *batch_shape, N_QUERIES, device=q.device, dtype=torch.float32
        )
        batch_size = 1
        for s in batch_shape:
            batch_size *= s

        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)

        flash_fwd_kernel[grid](
            q, k, v,
            o, l,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        ctx.save_for_backward(q, k, v, o, l)
        return o

    @staticmethod
    def backward(ctx, grad_O: Float32[Tensor, "..."]):
        raise NotImplementedError()
        

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # stride_qb, stride_kb ...
    # are stride of Q, K batch

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    max_S = tl.full((Q_TILE_SIZE,), float('-inf'), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    if is_causal:
        q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] >= k_indices[None, :]
            S = tl.where(mask, S, float(-1e6))

        new_max_S = tl.maximum(max_S, tl.max(S, axis=-1))
        P = tl.exp(S - new_max_S[:, None])
        l = tl.exp(max_S - new_max_S) * l + tl.sum(P, axis = -1)

        O = tl.exp(max_S - new_max_S)[:, None] * O + tl.dot(P, V) 
        max_S = new_max_S
        K_block_ptr = tl.advance(K_block_ptr, offsets=(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr, offsets=(K_TILE_SIZE,0))
    
    O = O / l[:, None]
    L = max_S + tl.log(l)
    tl.store(O_block_ptr, O, boundary_check=(0,1))
    tl.store(L_block_ptr, L, boundary_check=(0,))




        