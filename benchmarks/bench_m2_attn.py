"""M2 消融：同一个 ModelRunner、同一份权重和 KV 池，torch 参考 attention vs flashinfer 的单步耗时。

切后端的办法：把 runner.attn_backend 临时设成 None。prepare_* 看到 None 就不 plan、wrapper=None，
Attention 随之走 torch 的逐条 SDPA 循环。KV 写入两边都是 Triton store_kv，差别只在 attention。

计时口径是 runner.run 一整步：prepare（含 plan）+ 28 层前向 + lm_head + argmax.tolist()，
也就是引擎真正每步要付的代价。KV 内容是随机 token 的，不影响耗时。
运行（项目根目录）：python -m benchmarks.bench_m2_attn
"""
import random
import statistics
import time

import torch

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence
from mini_vllm.tests.runner_utils import build_runner

# 最大的 decode 配置是 128 条 × 2048 = 16384 块（bs=16），约 7.5 GB
NUM_BLOCKS = 20000
PREFILL_CASES = [(1, 512), (8, 512), (4, 2048)]        # (条数, 每条长度)，总 token <= 8192
DECODE_CASES = [(1, 512), (8, 512), (32, 512), (128, 512), (8, 2048), (32, 2048), (128, 2048)]
WARMUP, ITERS = 3, 10


def make_seqs(n: int, length: int, bm: BlockManager) -> list[Sequence]:
    seqs = [Sequence(i, [random.randrange(1000, 20000) for _ in range(length)]) for i in range(n)]
    for s in seqs:
        bm.allocate(s)
    return seqs


def time_step(runner, seqs, is_prefill) -> float:
    """毫秒，取中位数。run 末尾的 .tolist() 会同步 GPU，所以 perf_counter 量到的是真实耗时。"""
    for _ in range(WARMUP):
        runner.run(seqs, is_prefill)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        runner.run(seqs, is_prefill)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def bench(runner, cases, is_prefill):
    bs = runner.engine_cfg.block_size
    fi_backend = runner.attn_backend
    rows = []
    for n, length in cases:
        Sequence.block_size = bs
        bm = BlockManager(runner.engine_cfg.num_gpu_blocks, bs)
        # decode：序列长度 = 上下文，这一步的 token 写在最后一个 slot，反复写同一位置不影响计时
        seqs = make_seqs(n, length, bm)
        res = {}
        for name, backend in (("torch", None), ("flashinfer", fi_backend)):
            runner.attn_backend = backend
            res[name] = time_step(runner, seqs, is_prefill)
        runner.attn_backend = fi_backend
        for s in seqs:
            bm.deallocate(s)
        rows.append((n, length, res["torch"], res["flashinfer"]))
        print(f"  {'prefill' if is_prefill else 'decode '} {n:>3} 条 × {length:>4}: "
              f"torch {res['torch']:8.2f} ms   flashinfer {res['flashinfer']:7.2f} ms", flush=True)
    return rows


def main():
    random.seed(0)
    runner = build_runner(backend="flashinfer", num_gpu_blocks=NUM_BLOCKS)
    print(f"GPU: {torch.cuda.get_device_name()}  block_size={runner.engine_cfg.block_size}  "
          f"warmup={WARMUP} iters={ITERS}（中位数）")
    pre = bench(runner, PREFILL_CASES, is_prefill=True)
    dec = bench(runner, DECODE_CASES, is_prefill=False)

    print("\n| 阶段 | 条数 | 每条长度 | torch (ms) | flashinfer (ms) | 加速 | flashinfer 吞吐 (tok/s) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for n, L, t, f in pre:
        print(f"| prefill | {n} | {L} | {t:.2f} | {f:.2f} | {t / f:.1f}x | {n * L / f * 1000:,.0f} |")
    for n, L, t, f in dec:
        print(f"| decode | {n} | {L} | {t:.2f} | {f:.2f} | {t / f:.1f}x | {n / f * 1000:,.0f} |")


if __name__ == "__main__":
    main()
