# WeInfer Benchmark Results

## Setup

- **Model**: SmolLM-135M-Instruct-q0f32-MLC (0.3GB)
- **Prompt**: "Please introduce the Peking University in detail to me."
- **Max new tokens**: 32
- **xInterval (x)**: 4
- **Browser**: Chromium (Playwright)
- **Platform**: macOS (Darwin 25.2.0)

## Decode Speed (ms/token)

| Run | WebLLM Original | WebLLM +Cache | WebLLM +Cache+xInterval |
|-----|-----------------|---------------|-------------------------|
| 1   | 20.04           | 14.97         | 9.42                    |
| 2   | 18.81           | 15.07         | 9.22                    |
| 3   | 18.08           | 15.69         | 9.14                    |
| 4   | 20.19           | 16.55         | 9.14                    |
| 5   | 17.79           | 15.10         | 9.85                    |
| 6   | 19.42           | 15.34         | 9.71                    |
| 7   | 18.99           | 16.02         | 11.32                   |
| 8   | 17.58           | 17.52         | 10.53                   |
| 9   | 19.51           | 16.40         | 10.83                   |
| 10  | 18.02           | 16.39         | 10.97                   |
| **Avg** | **18.84**     | **15.90**     | **10.01**               |

## Summary

| Config                   | Avg (ms/token) | Speedup vs Original |
|--------------------------|----------------|---------------------|
| WebLLM Original          | 18.84          | 1.00x               |
| WebLLM +Cache            | 15.90          | 1.19x               |
| WebLLM +Cache+xInterval  | 10.01          | 1.88x               |

## Analysis

### WebLLM Original

Each decode step follows a synchronous flow:

```
forward() → GPU sync → allocate new buffer → sample token → process → repeat
```

Two bottlenecks limit performance:

1. **Per-token GPU buffer allocation**: `GPUSampler` creates a fresh bind group and uniform buffer for every token sample.
2. **Per-token GPU/CPU synchronization**: The CPU blocks waiting for the GPU after each token.

### WebLLM +Cache (1.19x speedup)

Introduces **buffer reuse** via the `GPUSamplerFixed2Dst` class:

- Pre-allocates GPU buffers once and reuses them across all decode steps.
- Eliminates per-token buffer allocation overhead.
- Still performs GPU/CPU sync after every single token.

The ~19% speedup comes purely from removing repeated GPU memory allocation.

### WebLLM +Cache+xInterval (1.88x speedup)

Adds **asynchronous pipelining** on top of buffer reuse. With `x=4` (default):

```
Step 1: forward → sample → store in slot[0] → NO SYNC
Step 2: forward → sample → store in slot[1] → NO SYNC
Step 3: forward → sample → store in slot[2] → NO SYNC
Step 4: forward → sample → store in slot[3] → SYNC (copy all 4 tokens at once)
```

Instead of blocking the CPU after every token, it only syncs every `x` tokens. This:

- **Reduces GPU/CPU synchronization overhead by 4x** (with x=4).
- **Keeps the GPU pipeline saturated** — the GPU doesn't idle waiting for CPU acknowledgment.
- **Batches the expensive `device.sync()` call** across multiple tokens.

### Optimization Impact

| Optimization             | What It Eliminates                | Speedup |
|--------------------------|-----------------------------------|---------|
| Buffer reuse (+Cache)    | Per-token GPU buffer allocation   | 1.19x   |
| Async pipeline (+xInterval) | Per-token GPU/CPU synchronization | 1.88x   |

The dominant bottleneck is **GPU/CPU synchronization**, not memory allocation. This is why the async pipeline — reducing sync frequency from every token to every 4 tokens — provides a much larger improvement than buffer reuse alone.
