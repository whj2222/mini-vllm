"""正确性验收：和 ref.json（HF greedy 生成）逐 token 对齐，走 ModelRunner 的 paged prefill 路径。

prompt + new_ids 拼成一条一次 prefill（teacher forcing），逐位置比较 top-1。
bf16 下完全一致做不到（打平的位置 argmax 会翻转），所以标准是：
总一致率 >= 98%，且不一致的位置 gap <= 0.25（logit 在 16~32 时约 2 个 bf16 刻度）。
基线（2026-10-12，M1 连续 cache）：2445/2475 = 98.79%，最大 gap 0.250。

运行（项目根目录）：python -m pytest mini_vllm/tests/test_ref_alignment.py -v -s
"""
import pytest
import torch

from mini_vllm.tests.runner_utils import (MAX_GAP, MIN_MATCH_RATE, build_runner, load_ref,
                                          mismatches, prefill_logits)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


@pytest.fixture(scope="module", params=["flashinfer", "torch"])
def runner(request):
    return build_runner(backend=request.param)


def test_ref_alignment(runner):
    ref = load_ref()
    total, bad = 0, []
    for i, r in enumerate(ref):
        b = mismatches(prefill_logits(runner, r["input_ids"], r["new_ids"]), r["new_ids"])
        total += len(r["new_ids"])
        bad += [(i, *x) for x in b]

    rate = (total - len(bad)) / total
    max_gap = max([0.0] + [b[-1] for b in bad])
    print(f"\n[{runner.engine_cfg.attn_backend}] prefill 路径 {total - len(bad)}/{total}"
          f"（{rate:.2%}），最大 gap {max_gap:.3f}")
    assert rate >= MIN_MATCH_RATE, f"一致率 {rate:.2%} 低于 {MIN_MATCH_RATE:.0%}"
    big = [b for b in bad if b[-1] > MAX_GAP]
    assert not big, f"gap 超过 {MAX_GAP} 的位置 (条, k, 我, 文件, gap)：{big}"
