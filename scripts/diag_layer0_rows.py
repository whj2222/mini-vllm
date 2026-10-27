"""接 diag_layer0：第 0 层误差不是"q 先乘 scale 再舍入"造成的（模拟误差比 SDPA 还小）。

新线索：第 0 层分数最大到 9337 / 24257（第 14 层只有 9）。新猜测：flashinfer 的 causal mask 用一个
有限的负数填被遮住的位置；如果某行所有合法分数都比这个数还低，被遮住的"未来 token"反而在 softmax
里胜出——这一行等于偷看了后面。

本脚本把第 0 层的误差拆到每个 (头, query 位置)：
- 列出误差最大的 10 行：合法分数的 max/min、可见 key 数、合法分数最大的 key 在哪
- 判据：flashinfer 那一行离"不加 causal 的 fp32 结果"比离"正确的 causal 结果"更近 → 漏 mask 坐实
- 统计误差 > 1% 的行数，以及这些行的合法 max 分数是否都很负
运行（项目根目录）：python -m scripts.diag_layer0_rows
"""
import math

import torch
import torch.nn.functional as F

from scripts.diag_flashinfer import OP_SEQS, capture_qkv, sdpa
from mini_vllm.tests.runner_utils import build_runner, load_ref


@torch.inference_mode()
def main():
    ref = load_ref()
    runner = build_runner(backend="flashinfer")
    for i in OP_SEQS:
        ids = ref[i]["input_ids"] + ref[i]["new_ids"]
        T = len(ids)
        wrapper = runner.attn_backend.plan_prefill(torch.tensor([0, T], dtype=torch.int32))
        q, k, v = capture_qkv(runner, ids)[0]                       # 第 0 层
        H, D = q.shape[1], q.shape[2]
        group = H // k.shape[1]
        scale = 1 / math.sqrt(D)

        gold = sdpa(q.float(), k.float(), v.float())                 # [T, H, D]
        fi = wrapper.run(q * q.shape[-1] ** -0.5, k, v).float()
        # 不加 causal 的 fp32 结果：如果 flashinfer 漏了 mask，它的错行会更像这个
        o_nc = F.scaled_dot_product_attention(q.float().transpose(0, 1)[None], k.float().transpose(0, 1)[None],
                                              v.float().transpose(0, 1)[None], enable_gqa=True)[0].transpose(0, 1)

        kk = k.float().repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q.float(), kk) * scale          # [H, T, T]
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        valid = scores.masked_fill(~mask, float("inf"))
        vmin = valid.amin(-1)                                                  # [H, T]
        vmax, vargmax = scores.masked_fill(~mask, float("-inf")).max(-1)

        row_err = (fi - gold).norm(dim=-1) / gold.norm(dim=-1)                # [T, H]
        d_causal = (fi - gold).norm(dim=-1)
        d_nocausal = (fi - o_nc).norm(dim=-1)

        print(f"== 第 {i} 条（T={T}）第 0 层：误差 > 1% 的行 {(row_err > 0.01).sum().item()}/{T * H} ==")
        print("   头  位置  行误差   合法max分数  合法min分数  argmax key  离causal  离no-causal")
        flat = row_err.flatten().topk(10)
        for e, idx in zip(flat.values.tolist(), flat.indices.tolist()):
            t, h = divmod(idx, H)
            print(f"   {h:2d}  {t:4d}  {e:7.3f}  {vmax[h, t].item():11.1f}  {vmin[h, t].item():11.1f}  "
                  f"{vargmax[h, t].item():9d}  {d_causal[t, h].item():8.3f}  {d_nocausal[t, h].item():10.3f}")


if __name__ == "__main__":
    main()
