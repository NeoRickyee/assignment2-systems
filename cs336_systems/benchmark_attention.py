import torch

DEVICE = torch.device("cuda")
SEQUENCE_LEN = [256, 1024, 4096, 8192, 16384]
EMBEDDING_DIM = [16, 32, 64, 128]

def benchmark_attention(
    attention_fn, batch_size, seq_len,
    num_heads, head_dim, dtype=torch.bfloat16
):
    # create dummy q, k, v
    q = torch.randn(batch_size, num_heads, seq_len,  head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len,  head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len,  head_dim, device=DEVICE, dtype=dtype, requires_grad=True)

    # dummy gradient for backward pass
    grad_output = torch.randn_like(q, device=DEVICE, dtype=dtype)

    # Warmup
    try:
        for _ in range(10):
            out = attention_fn(q, k, v)
            out.backward(grad_output)
    except torch.cuda.OutOfMemoryError:
        return None, None, None
    
    torch.cuda.synchronize()

    # Setup CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    num_iters = 100

    # Benchmark Forward pass
    try:
        start_event.record()
        for _ in range(num_iters):
            out = attention_fn(q, k, v)
        end_event.record()
    except torch.cuda.OutOfMemoryError:
        return None, None, None

    torch.cuda.synchronize()
    mem_used_bytes = torch.cuda.memory_allocated(DEVICE)
    mem_mb = mem_used_bytes / (1024 ** 2)
    forward_time_ms = start_event.elapsed_time(end_event) / num_iters

    # Benchmark Backward pass
    try:
        start_event.record()
        for _ in range(num_iters):
            q.grad, k.grad, v.grad = None, None, None
            out = attention_fn(q, k, v)
            out.backward(grad_output)
        end_event.record()
    except torch.cuda.OutOfMemoryError:
        return forward_time_ms, None, mem_mb

    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event) / num_iters
    backward_time_ms = total_time_ms - forward_time_ms

    return forward_time_ms, backward_time_ms, mem_mb
        
from cs336_basics.model import scaled_dot_product_attention
import pandas as pd

if __name__ == "__main__":
    # compiled_attention = torch.compile(scaled_dot_product_attention)
    compiled_attention = scaled_dot_product_attention
    results = []
    for seq_len in SEQUENCE_LEN:
        for d_model in EMBEDDING_DIM:
            fwd_ms, bwd_ms, mem_mb = benchmark_attention(
                compiled_attention, batch_size=8, seq_len=seq_len,
                num_heads=1, head_dim=d_model
            )
            results.append({
                "Seq Len": seq_len,
                "Head Dim": d_model,
                "Fwd Time (ms)": f"{fwd_ms:.3f}" if fwd_ms else "OOM",
                "Bwd Time (ms)": f"{bwd_ms:.3f}" if bwd_ms else "OOM",
                "Memory (MB)": f"{mem_mb:.2f}" if mem_mb else "OOM"
            })
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))