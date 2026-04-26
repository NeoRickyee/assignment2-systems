import torch
from torch import Tensor
import jaxtyping
from jaxtyping import Float32, Int32

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Float32[Tensor, "... N_QUERIES D"],
        k: Float32[Tensor, "... N_KEYS D"],
        v: Float32[Tensor, "... N_KEYS D"],
        is_causal=False,
    ):
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        q_size = q.size()
        k_size = k.size()

        ctx.D = q.size(-1)
        ctx.N_QUERIES = q_size[-2]
        ctx.N_KEYS = k_size[-2]
        assert ctx.D == k.size(-1) == v.size(-1), "D must match"
        
        batch_shape = q.shape[:-2]
        O_out = torch.empty_like(q)
        L_out = torch.empty(*batch_shape, ctx.N_QUERIES, device=q.device, dtype=q.dtype)
        ctx.scale = 1.0 / (ctx.D ** 0.5)

        
        for i in range(0, ctx.N_QUERIES, ctx.Q_TILE_SIZE):
            q_block: Float32[Tensor, "... Q_TILE_SIZE D"] = q[..., i:i+ctx.Q_TILE_SIZE, :]

            max_s: Float32[Tensor, "... Q_TILE_SIZE 1"] = \
                torch.full((*batch_shape, ctx.Q_TILE_SIZE, 1), float('-inf'), device=q.device)
            l: Float32[Tensor, "... Q_TILE_SIZE 1"] = \
                torch.zeros((*batch_shape, ctx.Q_TILE_SIZE, 1), device=q.device)
            O: Float32[Tensor, "... Q_TILE_SIZE D"] = \
                torch.zeros((*batch_shape, ctx.Q_TILE_SIZE, ctx.D), device=q.device)

            for j in range(0, ctx.N_KEYS, ctx.K_TILE_SIZE):
                k_block: Float32[Tensor, "... K_TILE_SIZE D"] = k[..., j:j+ctx.K_TILE_SIZE, :]
                v_block: Float32[Tensor, "... K_TILE_SIZE D"] = v[..., j:j+ctx.K_TILE_SIZE, :]

                s_block: Float32[Tensor, "... Q_TILE_SIZE K_TILE_SIZE"] = \
                    torch.matmul(q_block, torch.transpose(k_block, -1, -2)) * ctx.scale

                new_max_s: Float32[Tensor, "... Q_TILE_SIZE 1"] = \
                    torch.maximum(
                        max_s,
                        torch.max(s_block, dim=-1, keepdim=True).values
                    )
                
                p: Float32[Tensor, "... Q_TILE_SIZE K_TILE_SIZE"] = torch.exp(s_block - new_max_s)

                l = l * torch.exp(max_s - new_max_s) + torch.sum(p, dim=-1, keepdim=True)
                
                O = (torch.exp(max_s - new_max_s) * O + torch.matmul(p, v_block))
                max_s = new_max_s

            O = O / l
            O_out[..., i:i+ctx.Q_TILE_SIZE, :] = O
            
            L: Float32[Tensor, "... Q_TILE_SIZE 1"] = max_s + torch.log(l)
            L_out[..., i:i+ctx.Q_TILE_SIZE] = L.squeeze(-1)
        
        ctx.save_for_backward(q, k, v, O_out, L_out)
        return O_out
    
    @staticmethod
    def backward(ctx, grad_O: Float32[Tensor, "..."]):
        raise NotImplementedError()