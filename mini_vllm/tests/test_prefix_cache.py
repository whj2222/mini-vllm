"""M4 前缀缓存：命中真的发生、只算了新 token、结果和不命中时一致。

对照组：同一条序列在全新的 BlockManager 里整段 prefill（不可能命中），取每个位置的 logits。
每个场景开始前把 KV 池清零：命中路径要是读了没写过的块只会读到 0，不会"碰巧读到对的 KV"。
实验组：让它命中缓存，只 prefill 未命中的尾巴，走 paged prefill（has_cached=True）。
实验组每一行都应该和对照组同位置的那一行一致：
top-1 不同的位置 gap <= 0.25（bf16 打平），一致率 >= 98%。

三个场景：
1. 同一条跑两次，第一条还占着块（命中 ref_count > 0 的块）
2. 同一条跑两次，第一条已释放（命中空闲列表里的块）
3. 每条 ref 复制两份放进同一批 prefill：第二份命中第一份在【同一步】刚分到的块。
   这一步里那些块的 KV 还没算——靠的是每层先 store_kv 再做 attention。
   （ref 的 20 条 prompt 之间几乎没有公共前缀，所以要自己造共享。）

运行（项目根目录）：python -m pytest mini_vllm/tests/test_prefix_cache.py -v -s
"""
import pytest
import torch

from mini_vllm.engine import input_prep
from mini_vllm.engine.sequence import Sequence
from mini_vllm.tests.runner_utils import (MAX_GAP, MIN_MATCH_RATE, _block_manager, build_runner,
                                          forward_logits, load_ref)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


@pytest.fixture(scope="module", params=["flashinfer", "torch"])
def runner(request):
    return build_runner(backend=request.param)


def full_logits(runner, token_ids: list[int]) -> torch.Tensor:
    """对照组：全新池子整段 prefill，返回 [L, V]。"""
    bm = _block_manager(runner)
    seq = Sequence(0, token_ids)
    bm.allocate(seq)
    assert seq.num_cached_tokens == 0
    out = forward_logits(runner, [seq], is_prefill=True, all_positions=True)
    bm.deallocate(seq)
    return out


def compare(ref_rows: torch.Tensor, rows: torch.Tensor) -> tuple[int, list[tuple[int, float]]]:
    """逐行比 top-1，返回 (行数, [(行号, gap)])；gap = 对照组 top-1 logit - 实验组 top-1 在对照组里的 logit。"""
    assert ref_rows.shape == rows.shape
    a, b = ref_rows.argmax(-1), rows.argmax(-1)
    bad = (a != b).nonzero().flatten().tolist()
    return len(a), [(k, (ref_rows[k, a[k]] - ref_rows[k, b[k]]).item()) for k in bad]


def check(tag, n, bad):
    rate = (n - len(bad)) / n
    max_gap = max([0.0] + [g for _, g in bad])
    print(f"{tag} 一致 {n - len(bad)}/{n}（{rate:.2%}），最大 gap {max_gap:.3f}")
    assert rate >= MIN_MATCH_RATE, f"{tag} 一致率 {rate:.2%} 低于 {MIN_MATCH_RATE:.0%}"
    big = [x for x in bad if x[1] > MAX_GAP]
    assert not big, f"{tag} gap 超过 {MAX_GAP} 的位置 (行, gap)：{big}"


@pytest.mark.parametrize("keep_first", [True, False], ids=["first-alive", "first-freed"])
def test_same_prompt_twice(runner, keep_first):
    ref = load_ref()
    bs = runner.engine_cfg.block_size
    tag = f"\n[{runner.engine_cfg.attn_backend}/{'占用中' if keep_first else '已释放'}]"
    n_all, bad_all, cached_all = 0, [], 0
    for i, r in enumerate(ref):
        tokens = r["input_ids"] + r["new_ids"]
        L = len(tokens)
        expect_cached = (L - 1) // bs * bs

        runner.kv_cache.zero_()            # 池子清零：命中路径若读到没写过的块，只会读到 0，一定算错
        bm = _block_manager(runner)
        first = Sequence(0, tokens)
        bm.allocate(first)
        assert first.num_cached_tokens == 0
        base = forward_logits(runner, [first], is_prefill=True, all_positions=True)   # 对照组，同时写好 KV
        if not keep_first:
            bm.deallocate(first)
        second = Sequence(1, tokens)
        bm.allocate(second)

        assert second.num_cached_tokens == expect_cached, \
            f"第 {i} 条命中 {second.num_cached_tokens}，应为 {expect_cached}"
        shared = second.block_table[:expect_cached // bs]
        want_ref = 2 if keep_first else 1
        assert all(bm.blocks[b].ref_count == want_ref for b in shared), "命中块的引用计数不对"
        if expect_cached > 0:
            assert input_prep.prepare_prefill([second], bs).has_cached, "命中了却没走 paged prefill"

        rows = forward_logits(runner, [second], is_prefill=True, all_positions=True)
        assert rows.shape[0] == L - expect_cached, "送进模型的 token 数不等于未命中部分"
        n, bad = compare(base[expect_cached:], rows)
        n_all += n
        bad_all += [(i, k, g) for k, g in bad]
        cached_all += expect_cached

        bm.deallocate(second)
        if keep_first:
            bm.deallocate(first)
        assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks, "有块没还回去"

    total = sum(len(r["input_ids"]) + len(r["new_ids"]) for r in ref)
    print(f"{tag} 命中 {cached_all}/{total} 个 token，实际只算了 {n_all} 个")
    check(tag, n_all, [(f"{i}:{k}", g) for i, k, g in bad_all])


def test_same_batch_hits(runner):
    """每条 ref 复制两份放进同一批：第二份命中第一份同一步刚分到、KV 还没写的块。

    主判据：第二份 vs 同一次前向里第一份的同位置（同一批，批大小带来的 bf16 差异抵消），>= 98% 且 gap <= 0.25。
    辅助判据：vs 单跑整段只查 gap——同批 vs 单跑本来就有 GEMM 批大小造成的打平翻转（M2 起 ~30 处），
    和缓存无关，ref 的 prompt 之间也几乎没有公共前缀，不能拿一致率卡。"""
    ref = load_ref()
    bs = runner.engine_cfg.block_size
    tag = f"\n[{runner.engine_cfg.attn_backend}/同批互相命中]"
    runner.kv_cache.zero_()                # 清零才测得出"同一步先写后读"，不会读到之前留下的同样的 KV
    bm = _block_manager(runner)
    tokens = [r["input_ids"] + r["new_ids"] for r in ref]
    seqs = []
    for i, t in enumerate(tokens):
        seqs += [Sequence(2 * i, t), Sequence(2 * i + 1, t)]
    for s in seqs:
        bm.allocate(s)
    for i, t in enumerate(tokens):   # 第一份可能命中更早的 ref（17、18 条和前面共用第一块），不卡
        assert seqs[2 * i + 1].num_cached_tokens == (len(t) - 1) // bs * bs, f"第 {i} 条第二份命中数不对"

    rows = forward_logits(runner, seqs, is_prefill=True, all_positions=True)
    starts = [0]
    for s in seqs:
        starts.append(starts[-1] + len(s) - s.num_cached_tokens)
    assert rows.shape[0] == starts[-1], "送进模型的 token 数不等于各条未命中部分之和"

    n_pair, bad_pair, n_single, bad_single = 0, [], 0, []
    for i, t in enumerate(tokens):
        first = rows[starts[2 * i]:starts[2 * i + 1]]              # 位置 c1..L-1
        second = rows[starts[2 * i + 1]:starts[2 * i + 2]]         # 位置 c2..L-1，c2 >= c1
        c1, c2 = seqs[2 * i].num_cached_tokens, seqs[2 * i + 1].num_cached_tokens
        n, bad = compare(first[c2 - c1:], second)
        n_pair += n
        bad_pair += [(f"{i}:{k}", g) for k, g in bad]
        base = full_logits(runner, t)
        for part, off in ((first, c1), (second, c2)):
            n, bad = compare(base[off:], part)
            n_single += n
            bad_single += [(f"{i}:{k}", g) for k, g in bad]

    hit = sum(s.num_cached_tokens for s in seqs[1::2])   # deallocate 会把 num_cached_tokens 清零，先记下
    for s in seqs:
        bm.deallocate(s)
    assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks, "有块没还回去"
    print(f"{tag} 第二份共命中 {hit} 个 token，只算了 {n_pair} 个")
    max_single = max([0.0] + [g for _, g in bad_single])
    print(f"{tag} vs 单跑：不同 {len(bad_single)}/{n_single} 处，最大 gap {max_single:.3f}（只查 gap）")
    big = [x for x in bad_single if x[1] > MAX_GAP]
    assert not big, f"{tag} vs 单跑 gap 超过 {MAX_GAP}：{big}"
    check(tag + " 第二份 vs 第一份", n_pair, bad_pair)
