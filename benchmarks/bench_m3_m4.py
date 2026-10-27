"""M3 / M4 验收：离线吞吐（M3）、共享前缀下的 TTFT 与命中率（M4）。

主循环是测量用的最小版本（schedule -> runner.run 取 argmax -> postprocess），
等 engine/llm_engine.py 写好后换成 LLMEngine.step()，数字应该不变。

每种实现单独起一个进程跑（各自要占显存），在项目根目录：
  python -m benchmarks.bench_m3_m4 --mode mini --task throughput
  python -m benchmarks.bench_m3_m4 --mode hf   --task throughput
  python -m benchmarks.bench_m3_m4 --mode vllm --task throughput
  python -m benchmarks.bench_m3_m4 --mode mini --task ttft
结果追加到 benchmarks/results/m3_m4.jsonl。

负载（固定种子，三种实现完全相同）：
  throughput：256 条请求，输入长度 64~512 随机，输出固定 256（ignore_eos），总输出 65536 token。
  ttft：32 条请求 = 同一段 1024 token 的 system prompt + 各自 32 token 的问题。
        sequential：一条一条来（前一条做完再来下一条），量每条的首 token 延迟——最接近线上"共享 system prompt"。
        batch：32 条同时到，量每条从开始到出首 token 的时间。
        两种各跑一遍开缓存 / 关缓存（关 = 让 _match_prefix 永远返回空）。
"""
import argparse
import json
import os
import random
import statistics
import time

import torch

MODEL_PATH = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
RESULT_PATH = "benchmarks/results/m3_m4.jsonl"
EOS_IDS = {151645, 151643}
PAD_ID = 151643


def throughput_workload():
    rng = random.Random(0)
    prompts = [[rng.randrange(1000, 100000) for _ in range(rng.randint(64, 512))] for _ in range(256)]
    return prompts, 256


def ttft_workload():
    rng = random.Random(1)
    prefix = [rng.randrange(1000, 100000) for _ in range(1024)]
    return [prefix + [rng.randrange(1000, 100000) for _ in range(32)] for _ in range(32)]


def save(record):
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    record["gpu"] = torch.cuda.get_device_name()
    record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULT_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=1))


# ---------------- mini-vLLM ----------------

def mini_generate(runner, prompts, out_len, bm=None, prefix_cache=True):
    """跑完一组请求。返回 wall time、每条 TTFT、命中统计、抢占次数、步数。"""
    from mini_vllm.engine.block_manager import BlockManager
    from mini_vllm.engine.scheduler import Scheduler
    from mini_vllm.engine.sequence import Sequence

    bs = runner.engine_cfg.block_size
    Sequence.block_size = bs
    if bm is None:
        bm = BlockManager(runner.engine_cfg.num_gpu_blocks, bs)
    if not prefix_cache:
        bm._match_prefix = lambda seq: []
    sched = Scheduler(runner.engine_cfg, bm, EOS_IDS, runner.engine_cfg.max_model_len)
    seqs = [Sequence(i, p, max_tokens=out_len, ignore_eos=True) for i, p in enumerate(prompts)]
    for s in seqs:
        sched.add(s)

    ttft, first_cached = {}, {}
    n_prefill = n_decode = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not sched.is_finished():
        batch, is_prefill = sched.schedule()
        if is_prefill:
            n_prefill += 1
            for s in batch:
                first_cached.setdefault(s.seq_id, s.num_cached_tokens)   # 只记第一次上车，重算不算
        else:
            n_decode += 1
        tokens = runner.run(batch, is_prefill)          # .tolist() 里已经同步
        now = time.perf_counter() - t0
        if is_prefill:
            for s in batch:
                ttft.setdefault(s.seq_id, now)
        sched.postprocess(batch, tokens)
    wall = time.perf_counter() - t0
    assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks, "有块没还回去"   # 缓存块 ref=0 也在空闲列表里
    return dict(wall=wall, ttft=[ttft[i] for i in range(len(prompts))],
                cached=sum(first_cached.values()), prompt_tokens=sum(map(len, prompts)),
                preemptions=sched.num_preemptions, prefill_steps=n_prefill, decode_steps=n_decode)


def build_mini():
    from mini_vllm.config import EngineConfig
    from mini_vllm.engine.model_runner import ModelRunner
    runner = ModelRunner(MODEL_PATH, EngineConfig())
    mini_generate(runner, throughput_workload()[0][:8], 8)                  # 预热：CUDA 初始化、flashinfer JIT
    p = ttft_workload()[:2]
    mini_generate(runner, p, 2)                                             # 预热 paged prefill（第二条命中）
    return runner


def mini_throughput():
    runner = build_mini()
    prompts, out_len = throughput_workload()
    r = mini_generate(runner, prompts, out_len)
    out = len(prompts) * out_len
    save(dict(mode="mini", task="throughput", backend=runner.engine_cfg.attn_backend,
              num_gpu_blocks=runner.engine_cfg.num_gpu_blocks, requests=len(prompts), output_tokens=out,
              wall_s=round(r["wall"], 3), output_tok_per_s=round(out / r["wall"], 1),
              total_tok_per_s=round((out + r["prompt_tokens"]) / r["wall"], 1),
              preemptions=r["preemptions"], prefill_steps=r["prefill_steps"], decode_steps=r["decode_steps"]))


def summarize(ttft):
    ms = sorted(t * 1000 for t in ttft)
    return dict(mean_ms=round(statistics.mean(ms), 2), p50_ms=round(ms[len(ms) // 2], 2),
                p90_ms=round(ms[int(len(ms) * 0.9)], 2), max_ms=round(ms[-1], 2))


def mini_ttft():
    from mini_vllm.engine.block_manager import BlockManager
    runner = build_mini()
    prompts = ttft_workload()
    bs = runner.engine_cfg.block_size
    for prefix_cache in (False, True):
        # sequential：同一个 BlockManager 贯穿始终，前一条留下的块后一条才能命中
        bm = BlockManager(runner.engine_cfg.num_gpu_blocks, bs)
        ttft, cached = [], 0
        for p in prompts:
            r = mini_generate(runner, [p], 1, bm=bm, prefix_cache=prefix_cache)
            ttft.append(r["ttft"][0])
            cached += r["cached"]
        total = sum(map(len, prompts))
        save(dict(mode="mini", task="ttft", arrival="sequential", prefix_cache=prefix_cache,
                  requests=len(prompts), prompt_len=len(prompts[0]), hit_rate=round(cached / total, 4),
                  **summarize(ttft), first_request_ms=round(ttft[0] * 1000, 2),
                  rest_mean_ms=round(statistics.mean(ttft[1:]) * 1000, 2)))

        # batch：32 条同时到
        r = mini_generate(runner, prompts, 16, prefix_cache=prefix_cache)
        save(dict(mode="mini", task="ttft", arrival="batch", prefix_cache=prefix_cache,
                  requests=len(prompts), prompt_len=len(prompts[0]),
                  hit_rate=round(r["cached"] / r["prompt_tokens"], 4), **summarize(r["ttft"]),
                  prefill_steps=r["prefill_steps"], wall_s=round(r["wall"], 3)))


# ---------------- HF transformers ----------------

def hf_throughput(batch_size):
    from transformers import AutoModelForCausalLM
    from mini_vllm.config import resolve_model_path
    model = AutoModelForCausalLM.from_pretrained(resolve_model_path(MODEL_PATH), dtype=torch.bfloat16).cuda().eval()
    prompts, out_len = throughput_workload()

    def run(group, n_new):
        width = max(map(len, group))
        ids = torch.tensor([[PAD_ID] * (width - len(p)) + p for p in group], device="cuda")        # 左 padding
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in group], device="cuda")
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_new, min_new_tokens=n_new,
                             do_sample=False, pad_token_id=PAD_ID)
        assert out.shape[1] - width == n_new

    with torch.inference_mode():
        run(prompts[:4], 8)                                                  # 预热
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(prompts), batch_size):
            run(prompts[i:i + batch_size], out_len)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    out = len(prompts) * out_len
    save(dict(mode="hf", task="throughput", batch_size=batch_size, requests=len(prompts), output_tokens=out,
              wall_s=round(wall, 3), output_tok_per_s=round(out / wall, 1),
              total_tok_per_s=round((out + sum(map(len, prompts))) / wall, 1)))


# ---------------- vLLM ----------------

def vllm_throughput(enforce_eager):
    from vllm import LLM, SamplingParams
    from mini_vllm.config import resolve_model_path
    llm = LLM(model=resolve_model_path(MODEL_PATH), dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, enforce_eager=enforce_eager, seed=0)
    prompts, out_len = throughput_workload()
    sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0)
    llm.generate([{"prompt_token_ids": p} for p in prompts[:8]], SamplingParams(max_tokens=8, ignore_eos=True))
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    out = sum(len(o.outputs[0].token_ids) for o in outs)
    assert out == len(prompts) * out_len
    save(dict(mode="vllm", task="throughput", enforce_eager=enforce_eager, requests=len(prompts),
              output_tokens=out, wall_s=round(wall, 3), output_tok_per_s=round(out / wall, 1),
              total_tok_per_s=round((out + sum(map(len, prompts))) / wall, 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mini", "hf", "vllm"], required=True)
    ap.add_argument("--task", choices=["throughput", "ttft"], default="throughput")
    ap.add_argument("--hf-batch", type=int, default=64)
    ap.add_argument("--eager", action="store_true", help="vLLM 关 CUDA Graph（和 M5 之前的 mini-vLLM 同条件）")
    args = ap.parse_args()
    if args.mode == "mini":
        mini_throughput() if args.task == "throughput" else mini_ttft()
    elif args.mode == "hf":
        hf_throughput(args.hf_batch)
    else:
        vllm_throughput(args.eager)
