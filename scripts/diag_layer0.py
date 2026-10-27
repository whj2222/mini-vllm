"""接 diag_flashinfer：flashinfer 只在第 0 层误差是 SDPA 的 40~65 倍，其余 27 层 1.0x。为什么？

猜测：Qwen2.5 第 0 层 q/k（带 bias）数值特别大，attention 分数动辄上百，softmax 极尖。
flashinfer 若先把 q 乘上 sm_scale（或 sm_scale·log2e）再舍入回 bf16 才进 MMA，
分数的绝对误差会跟着分数一起放大；SDPA 是在 fp32 分数上乘 scale，没有这次舍入。

本脚本对第 0 层和一个普通层（第 14 层）打印：
- q/k 的最大绝对值、attention 分数（已乘 scale）的最大值
- 四种实现相对 fp32 的误差：bf16 SDPA / flashinfer / "q 先乘 scale 再舍入 bf16"的 SDPA 模拟 /
  "q 先乘 scale·log2e 再舍入"的模拟
如果某个模拟版的误差和 flashinfer 同量级，猜测就坐实了。
运行（项目根目录）：python -m scripts.diag_layer0
"""
import math

import torch

from scripts.diag_flashinfer import OP_SEQS, capture_qkv, sdpa
from mini_vllm.tests.runner_utils import build_runner, load_ref

LAYERS = [0, 14]


def rel(o, gold):
    return ((o.float() - gold).norm() / gold.norm()).item()


@torch.inference_mode()
def main():
    ref = load_ref()
    runner = build_runner(backend="flashinfer")
    for i in OP_SEQS:
        ids = ref[i]["input_ids"] + ref[i]["new_ids"]
        wrapper = runner.attn_backend.plan_prefill(torch.tensor([0, len(ids)], dtype=torch.int32))
        rec = capture_qkv(runner, ids)
        for layer in LAYERS:
            q, k, v = rec[layer]
            D = q.shape[-1]
            scale = 1 / math.sqrt(D)
            gold = sdpa(q.float(), k.float(), v.float())

            # 分数 = q·k * scale，GQA：q 头 h 对应 kv 头 h // (H / Hkv)
            group = q.shape[1] // k.shape[1]
            kk = k.float().repeat_interleave(group, dim=1)
            scores = torch.einsum("thd,shd->hts", q.float(), kk) * scale
            mask = torch.ones(len(ids), len(ids), dtype=torch.bool, device=q.device).tril()
            smax = scores.masked_fill(~mask, float("-inf")).amax().item()

            q_pre = (q.float() * scale).to(torch.bfloat16)                  # 模拟：先乘 scale 再舍入
            q_pre2 = (q.float() * scale * math.log2(math.e)).to(torch.bfloat16)  # 模拟：再折进 log2e
            # 用 fp32 算后面的部分，只保留"q 舍入"这一个误差来源
            o_pre = sdpa(q_pre.float() / scale, k.float(), v.float())
            o_pre2 = sdpa(q_pre2.float() / (scale * math.log2(math.e)), k.float(), v.float())

            print(f"[{i:02d}] 第 {layer:2d} 层  max|q| {q.abs().max().item():7.1f}  "
                  f"max|k| {k.abs().max().item():7.1f}  max 分数 {smax:7.1f}")
            print(f"      相对误差  SDPA {rel(sdpa(q, k, v), gold):.2e}  "
                  f"flashinfer {rel(wrapper.run(q * q.shape[-1] ** -0.5, k, v), gold):.2e}  "
                  f"模拟 q·scale {rel(o_pre, gold):.2e}  模拟 q·scale·log2e {rel(o_pre2, gold):.2e}")


if __name__ == "__main__":
    main()
