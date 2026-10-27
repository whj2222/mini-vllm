"""decode 路径验收：逐 token 读写 paged KV 池，而不是一次 prefill 算完。

test_ref_alignment 把 prompt + new_ids 一次 prefill，池子只写不读，decode 从没执行过。这里补四项：
1. 单条 decode vs 一次 prefill：同一 runner，只差"KV 从池子按 block 读回"这一个变量，应几乎一致。
2. 单条 decode vs ref.json：标准和 test_ref_alignment 相同（>=98%，gap <= 0.25）。
3. 自由 greedy：第一个分叉点之前和 ref 完全一致，且分叉点在 decode 路径上是近似打平。
4. 20 条同批 decode：vs ref 达标；和单跑 top-1 不同的位置，必须在同批 logits 上也是打平
   （gap <= 0.25）——这条证明同批 vs 单跑的差异来自 bf16 非批不变性，不是串台。

每项跑三组：flashinfer bs=16 / flashinfer bs=1（每 token 一块，slot 算错最容易暴露）/ torch 参考后端。
运行（项目根目录）：python -m pytest mini_vllm/tests/test_decode_path.py -v -s
只跑一组：-k flashinfer-bs16
"""
import pytest
import torch

from mini_vllm.tests.runner_utils import (MAX_GAP, MIN_MATCH_RATE, build_runner, greedy_generate,
                                          load_ref, mismatches, prefill_logits, teacher_forcing)

# decode vs prefill 的 logits 最大绝对差，只用来兜住严重错误。
# 实测（2026-10-12，A4000，连续 cache + SDPA）：最大 1.688，|logit| 约 40 时 bf16 一格 0.25，
# 来自 prefill（T 个 query）和 decode（1 个 query）的 attention 分块不同、28 层累积。
# 池子读错时这里会到几到几十，并且会出现非打平的 top-1 翻转，后者才是主判据。
MAX_PREFILL_DIFF = 3.0

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")

CONFIGS = [("flashinfer", 16), ("flashinfer", 1), ("torch", 16)]


@pytest.fixture(scope="module", params=CONFIGS, ids=[f"{b}-bs{s}" for b, s in CONFIGS])
def runner(request):
    backend, block_size = request.param
    return build_runner(backend=backend, block_size=block_size)


@pytest.fixture(scope="module")
def ref():
    return load_ref()


@pytest.fixture(scope="module")
def single(runner, ref):
    """每条单独跑 decode teacher forcing 和一次 prefill，只留下各测试要用的标量（全量 logits 太大）。"""
    recs = []
    for r in ref:
        new = r["new_ids"]
        D = teacher_forcing(runner, [r["input_ids"]], [new])[0]      # [N, V]
        P = prefill_logits(runner, r["input_ids"], new)              # [N, V]
        dtop, ptop = D.argmax(-1), P.argmax(-1)
        idx = torch.arange(len(new), device=D.device)
        tgt = torch.tensor(new, device=D.device)
        recs.append(dict(
            dtop=dtop.tolist(),
            dmax=D[idx, dtop].tolist(),          # decode 路径 top-1 的 logit
            dref=D[idx, tgt].tolist(),           # decode 路径上参考 token 的 logit
            maxdiff=(D - P).abs().max().item(),
            # decode 与 prefill top-1 不同的位置，在 prefill logits 上两者差多少
            flips=[(k, (P[k, ptop[k]] - P[k, dtop[k]]).item())
                   for k in (dtop != ptop).nonzero().flatten().tolist()],
        ))
    return recs


def test_decode_matches_prefill(single):
    worst = max(s["maxdiff"] for s in single)
    flips = [(i, k, g) for i, s in enumerate(single) for k, g in s["flips"] if g > MAX_GAP]
    n_flip = sum(len(s["flips"]) for s in single)
    print(f"\nmax|decode-prefill| {worst:.3f}，top-1 翻转 {n_flip} 处（非打平 {len(flips)}）")
    # 先查主判据，这样两项都失败时能先看到翻转明细
    assert not flips, f"非打平位置 top-1 不同 (条, k, gap)：{flips}"
    assert worst <= MAX_PREFILL_DIFF, f"decode 与 prefill logits 最大差 {worst:.3f}，池子读写可能有误"


def test_decode_vs_ref(ref, single):
    total, bad = 0, []
    for i, (r, s) in enumerate(zip(ref, single)):
        total += len(r["new_ids"])
        bad += [(i, k, m, t, s["dmax"][k] - s["dref"][k])
                for k, (m, t) in enumerate(zip(s["dtop"], r["new_ids"])) if m != t]
    rate = (total - len(bad)) / total
    max_gap = max([0.0] + [b[-1] for b in bad])
    print(f"\n单条 decode {total - len(bad)}/{total}（{rate:.2%}），最大 gap {max_gap:.3f}")
    assert rate >= MIN_MATCH_RATE, f"一致率 {rate:.2%} 低于 {MIN_MATCH_RATE:.0%}"
    big = [b for b in bad if b[-1] > MAX_GAP]
    assert not big, f"gap 超过 {MAX_GAP} 的位置 (条, k, 我, 文件, gap)：{big}"


def test_greedy_free_run(runner, ref, single):
    problems, n_exact = [], 0
    for i, (r, s) in enumerate(zip(ref, single)):
        new = r["new_ids"]
        out = greedy_generate(runner, r["input_ids"], len(new))
        k = next((j for j, (a, b) in enumerate(zip(out, new)) if a != b), None)
        if k is None:
            if len(out) != len(new):
                problems.append((i, "长度不同", len(out), len(new)))
            else:
                n_exact += 1
            continue
        # 分叉点之前输入完全相同，自由生成的 token 必须等于单条 teacher forcing 的 argmax
        if out[k] != s["dtop"][k]:
            problems.append((i, k, "自由生成与 teacher forcing argmax 不同", out[k], s["dtop"][k]))
            continue
        gap = s["dmax"][k] - s["dref"][k]
        print(f"\n[{i:02d}] 第 {k} 个 token 分叉（我 {out[k]} / 文件 {new[k]}，gap {gap:.3f}）", end="")
        if gap > MAX_GAP:
            problems.append((i, k, "分叉点不是打平", gap))
    print(f"\n自由生成完全一致 {n_exact}/{len(ref)} 条")
    assert not problems, f"问题：{problems}"


def test_batch(runner, ref, single):
    batch = teacher_forcing(runner, [r["input_ids"] for r in ref], [r["new_ids"] for r in ref])

    total, bad_ref, vs_single = 0, [], []
    for i, (r, s, B) in enumerate(zip(ref, single, batch)):
        new = r["new_ids"]
        total += len(new)
        bad_ref += [(i, *x) for x in mismatches(B, new)]
        btop = B.argmax(-1).tolist()
        # 同批和单跑 top-1 不同：在同批 logits 上，同批 top-1 比单跑 top-1 高多少
        vs_single += [(i, k, (B[k, b] - B[k, a]).item())
                      for k, (b, a) in enumerate(zip(btop, s["dtop"])) if b != a]

    rate = (total - len(bad_ref)) / total
    max_ref_gap = max([0.0] + [b[-1] for b in bad_ref])
    max_single_gap = max([0.0] + [g for *_, g in vs_single])
    print(f"\n20 条同批 vs ref {total - len(bad_ref)}/{total}（{rate:.2%}），最大 gap {max_ref_gap:.3f}")
    print(f"同批 vs 单跑 不同 {len(vs_single)} 处，最大 gap {max_single_gap:.3f}")

    assert rate >= MIN_MATCH_RATE, f"同批一致率 {rate:.2%} 低于 {MIN_MATCH_RATE:.0%}"
    big = [b for b in bad_ref if b[-1] > MAX_GAP]
    assert not big, f"同批 vs ref gap 超过 {MAX_GAP} (条, k, 我, 文件, gap)：{big}"
    big = [x for x in vs_single if x[-1] > MAX_GAP]
    assert not big, f"同批 vs 单跑 非打平差异，疑似串台 (条, k, gap)：{big}"
