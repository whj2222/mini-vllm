"""测试和脚本共用：通过 ModelRunner 走 paged 路径，拿每个位置的 logits。

只调用 ModelRunner.prepare_prefill / prepare_decode 和 runner.model，不改 ModelRunner。
和 ModelRunner.run 的唯一区别：返回 float32 logits 而不是 argmax，这样才能算 gap
（gap = 我的 top-1 logit - 对照 token 的 logit；bf16 打平时 gap 只有一两个刻度）。
"""
import json

import torch

from mini_vllm.config import EngineConfig
from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.model_runner import ModelRunner
from mini_vllm.engine.sequence import Sequence

MODEL_PATH = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
REF_PATH = "scripts/ref.json"          # 在项目根目录运行
MIN_MATCH_RATE = 0.98
MAX_GAP = 0.25                         # logit 16~32 时约 2 个 bf16 刻度
QWEN2_EOS_IDS = {151645, 151643}
# 测试固定池子大小，不走 profile：profile 会吃掉 90% 显存，别的 fixture 就放不下了。
# 8192 块：bs=16 时 13 万 token（3.8 GB），bs=1 时 8192 token，20 条 ref 同批共约 3100 token，都够。
TEST_NUM_BLOCKS = 8192


def load_ref() -> list[dict]:
    with open(REF_PATH) as f:
        return json.load(f)


def build_runner(backend: str = "flashinfer", block_size: int = 16,
                 num_gpu_blocks: int = TEST_NUM_BLOCKS) -> ModelRunner:
    return ModelRunner(MODEL_PATH, EngineConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks,
                                                attn_backend=backend))


def _block_manager(runner: ModelRunner) -> BlockManager:
    bs = runner.engine_cfg.block_size
    Sequence.block_size = bs           # Sequence.num_blocks 读的是类属性，必须和引擎一致
    return BlockManager(runner.engine_cfg.num_gpu_blocks, bs)


@torch.inference_mode()
def forward_logits(runner: ModelRunner, seqs: list[Sequence], is_prefill: bool,
                   all_positions: bool = False) -> torch.Tensor:
    """一步前向。prefill 默认只取每条最后一个位置 [B, V]；all_positions=True 返回 [T, V]。"""
    prepare = runner.prepare_prefill if is_prefill else runner.prepare_decode
    input_ids, positions, md = prepare(seqs)
    hidden = runner.model.model(input_ids, positions, md)
    if is_prefill and not all_positions:
        hidden = hidden[(md.cu_seqlens[1:] - 1).long()]
    return runner.model.lm_head(hidden).float()


@torch.inference_mode()
def teacher_forcing(runner: ModelRunner, prompts: list[list[int]],
                    news: list[list[int]]) -> list[torch.Tensor]:
    """同批 prefill，之后每步喂 news 里的 token 做 decode。
    返回每条的 [N_i, V]，第 k 行预测 news[i][k]。长度不同的条陆续退出，batch 越来越小。"""
    bm = _block_manager(runner)
    seqs = [Sequence(i, p) for i, p in enumerate(prompts)]
    for s in seqs:
        assert bm.can_allocate(s)
        bm.allocate(s)

    rows = [[r] for r in forward_logits(runner, seqs, is_prefill=True)]
    for k in range(max(len(n) for n in news) - 1):
        active = [s for s in seqs if k + 1 < len(news[s.seq_id])]
        for s in active:
            s.append_token(news[s.seq_id][k])
            assert bm.can_append(s)
            bm.may_append(s)
        for s, r in zip(active, forward_logits(runner, active, is_prefill=False)):
            rows[s.seq_id].append(r)

    for s in seqs:
        bm.deallocate(s)
    assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks, "有块没还回去"
    return [torch.stack(r) for r in rows]


@torch.inference_mode()
def prefill_logits(runner: ModelRunner, prompt: list[int], new: list[int]) -> torch.Tensor:
    """prompt + new 拼成一条，一次 prefill 算完，返回 [N, V]，第 k 行预测 new[k]。"""
    bm = _block_manager(runner)
    seq = Sequence(0, prompt + new)
    bm.allocate(seq)
    logits = forward_logits(runner, [seq], is_prefill=True, all_positions=True)
    bm.deallocate(seq)
    T = len(prompt)
    return logits[T - 1: T - 1 + len(new)]


@torch.inference_mode()
def greedy_generate(runner: ModelRunner, prompt: list[int], max_new_tokens: int,
                    eos_ids: set[int] = QWEN2_EOS_IDS) -> list[int]:
    """单条自由 greedy 生成（接替已删除的 engine/naive.py），遇到 eos 停，eos 本身也返回。"""
    bm = _block_manager(runner)
    seq = Sequence(0, prompt)
    bm.allocate(seq)
    out = runner.run([seq], is_prefill=True)
    while out[-1] not in eos_ids and len(out) < max_new_tokens:
        seq.append_token(out[-1])
        assert bm.can_append(seq)
        bm.may_append(seq)
        out += runner.run([seq], is_prefill=False)
    bm.deallocate(seq)
    return out


def mismatches(pred: torch.Tensor, new_ids: list[int]) -> list[tuple[int, int, int, float]]:
    """(k, 我的 top-1, 参考 token, gap)，gap = 我的 top-1 logit - 参考 token 的 logit。"""
    mine = pred.argmax(-1).tolist()
    return [(k, m, t, (pred[k, m] - pred[k, t]).item())
            for k, (m, t) in enumerate(zip(mine, new_ids)) if m != t]
