"""M3 验收：KV 池调小触发 recompute 抢占，输出仍和 ref.json 对齐。

Scheduler + ModelRunner 跑真实主循环（schedule -> 前向 -> postprocess），只把喂回去的 token
换成 ref 的 new_ids（teacher forcing）：自由生成在 bf16 打平处会分叉，没法逐 token 比；
teacher forcing 下每一步的输入都确定，抢不抢占预测都应该一样。

被抢的序列重新上车时，prompt + 已生成部分整段重新 prefill，最后一个位置预测下一个 token——
这一行就是 recompute 的检验：KV 重算错了（slot 写错块、position 没从 0 开始、读到老块残留），
这里的预测会偏。

池子大小 = max(最长一条要的块数, 全部同时在车上要的块数 // 4)，保证一定抢、且不会死循环。
标准同 test_ref_alignment：一致率 >= 98%，不一致处 gap <= 0.25。

运行（项目根目录）：python -m pytest mini_vllm/tests/test_preemption.py -v -s
"""
import pytest
import torch

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence
from mini_vllm.tests.runner_utils import (MAX_GAP, MIN_MATCH_RATE, QWEN2_EOS_IDS, build_runner,
                                          forward_logits, load_ref)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")

BLOCK_SIZE = 16
MAX_STEPS = 100_000                    # 调度写坏陷入死循环时及时报错


def _num_blocks(ref: list[dict]) -> int:
    need = [-(-(len(r["input_ids"]) + len(r["new_ids"])) // BLOCK_SIZE) for r in ref]
    return max(max(need), sum(need) // 4)


@pytest.fixture(scope="module", params=["flashinfer", "torch"])
def runner(request):
    return build_runner(backend=request.param, num_gpu_blocks=_num_blocks(load_ref()))


@torch.inference_mode()
def run_engine(runner, ref):
    """返回 (scheduler, bm, bad, n_pred, recomputes)。
    bad: 不一致位置 (条, k, 我, 参考, gap)；n_pred[i]: 第 i 条得到的预测数；
    recomputes: 重新 prefill 事件 (条, 当时已生成数)。"""
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(runner.engine_cfg.num_gpu_blocks, BLOCK_SIZE)
    scheduler = Scheduler(runner.engine_cfg, bm, QWEN2_EOS_IDS, runner.engine_cfg.max_model_len)
    news = [r["new_ids"] for r in ref]
    for i, r in enumerate(ref):
        scheduler.add(Sequence(i, r["input_ids"], max_tokens=len(r["new_ids"]), ignore_eos=True))

    bad, n_pred, recomputes = [], [0] * len(ref), []
    for _ in range(MAX_STEPS):
        if scheduler.is_finished():
            break
        batch, is_prefill = scheduler.schedule()
        logits = forward_logits(runner, batch, is_prefill)
        feed = []
        for s, row in zip(batch, logits):
            k = s.num_completion_tokens        # 这一行预测 news[i][k]
            if is_prefill and k > 0:
                recomputes.append((s.seq_id, k))
            t = news[s.seq_id][k]
            m = row.argmax().item()
            if m != t:
                bad.append((s.seq_id, k, m, t, (row[m] - row[t]).item()))
            n_pred[s.seq_id] += 1
            feed.append(t)
        scheduler.postprocess(batch, feed)
    else:
        pytest.fail(f"{MAX_STEPS} 步还没跑完，调度可能死循环")
    return scheduler, bm, bad, n_pred, recomputes


def test_preemption_matches_ref(runner):
    ref = load_ref()
    scheduler, bm, bad, n_pred, recomputes = run_engine(runner, ref)

    total = sum(len(r["new_ids"]) for r in ref)
    rate = (total - len(bad)) / total
    max_gap = max([0.0] + [b[-1] for b in bad])
    re_keys = set(recomputes)
    re_bad = [b for b in bad if (b[0], b[1]) in re_keys]
    print(f"\n[{runner.engine_cfg.attn_backend}] 池 {runner.engine_cfg.num_gpu_blocks} 块，"
          f"抢占 {scheduler.num_preemptions} 次，重新 prefill {len(recomputes)} 次")
    print(f"一致 {total - len(bad)}/{total}（{rate:.2%}），最大 gap {max_gap:.3f}；"
          f"重新 prefill 那一步不一致 {len(re_bad)}/{len(recomputes)}")

    assert scheduler.num_preemptions > 0, "没触发抢占，池子不够小"
    assert recomputes, "抢占了但没有序列重新 prefill"
    assert n_pred == [len(r["new_ids"]) for r in ref], "有序列预测数不对（丢 token 或多跑）"
    assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks, "有块没还回去"
    assert rate >= MIN_MATCH_RATE, f"一致率 {rate:.2%} 低于 {MIN_MATCH_RATE:.0%}"
    big = [b for b in bad if b[-1] > MAX_GAP]
    assert not big, f"gap 超过 {MAX_GAP} 的位置 (条, k, 我, 参考, gap)：{big}"
