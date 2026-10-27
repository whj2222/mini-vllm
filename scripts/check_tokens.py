"""逐 token 对比 mini-vllm 和 ref.json（HF greedy 生成），打印每条的明细。

teacher forcing：上文强制用文件里的 token，看每个位置我的 top-1 是不是文件里那个。
走 ModelRunner 的 paged 路径（prompt + new 一次 prefill），判定标准见 tests/test_ref_alignment.py。
运行（项目根目录）：PYTHONPATH=. python scripts/check_tokens.py [flashinfer|torch]
"""
import sys

from mini_vllm.tests.runner_utils import build_runner, load_ref, mismatches, prefill_logits


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "flashinfer"
    runner = build_runner(backend=backend)

    total = n_bad = 0
    max_gap = 0.0
    for i, item in enumerate(load_ref()):
        bad = mismatches(prefill_logits(runner, item["input_ids"], item["new_ids"]), item["new_ids"])
        N = len(item["new_ids"])
        total += N
        n_bad += len(bad)
        detail = "  ".join(f"#{k}(我{m}/文件{t} gap {g:.3f})" for k, m, t, g in bad)
        print(f"[{i:02d}] 一致 {N - len(bad)}/{N}  {detail}")
        max_gap = max([max_gap] + [g for *_, g in bad])

    print(f"[{backend}] 总计 {total - n_bad}/{total}（{(total - n_bad) / total:.2%}），最大 gap {max_gap:.3f}")


if __name__ == "__main__":
    main()
