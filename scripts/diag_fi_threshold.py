"""接 diag_layer0_rows：第 0 层只有 3 行（第 6 条）/ 2 行（第 16 条）错，行误差恰好 1.000，
全在头 3，且这些行的"合法分数全都极负"（max 约 -5374 / -4618）。其余行误差 0.3% 左右，正常。

行误差 = 1.000 意味着 ||fi - gold|| = ||gold||，最像 flashinfer 把这一行输出成了 0
（当成"整行被 mask"处理了）。本脚本：
A. 真实数据：打印这几行 flashinfer 输出的范数，以及它离 0 / 离"未来 token 的 v 均值"各多远
B. 合成数据扫阈值：所有 key 相同，让每行分数都等于 s（softmax 应是均匀分布，结果 = v 的均值），
   s 从 -10 扫到 -20000，分别测 ragged prefill（causal / 非 causal）和 paged decode。
   找出从哪个 s 开始坏，以及 decode 路径是否同样中招——可直接作为给 flashinfer 提 issue 的最小复现。
运行（项目根目录）：python -m scripts.diag_fi_threshold
"""
import math

import torch

from scripts.diag_flashinfer import capture_qkv, sdpa
from mini_vllm.tests.runner_utils import build_runner, load_ref

H, HKV, D, BS = 12, 2, 128, 16
SCALE = 1 / math.sqrt(D)
# 新用法下阈值移到缩放后分数 -5e4，所以扫到 -80000 看它在哪儿坏
SWEEP = [-10, -1000, -4000, -5000, -8000, -20000, -40000, -50000, -60000, -80000]
BAD_ROWS = {6: [(3, 8), (3, 78), (3, 5)], 16: [(3, 127), (3, 82)]}     # (头, 位置)，来自 diag_layer0_rows


@torch.inference_mode()
def real_rows(runner, ref):
    print("== A. 真实数据里出错的行 ==")
    for i, rows in BAD_ROWS.items():
        ids = ref[i]["input_ids"] + ref[i]["new_ids"]
        T = len(ids)
        wrapper = runner.attn_backend.plan_prefill(torch.tensor([0, T], dtype=torch.int32))
        q, k, v = capture_qkv(runner, ids)[0]
        gold = sdpa(q.float(), k.float(), v.float())
        fi = wrapper.run(q * q.shape[-1] ** -0.5, k, v).float()
        group = H // HKV
        for h, t in rows:
            vf = v[:, h // group].float()                                   # 这个头对应的 kv 头
            fut = vf[t + 1:].mean(0) if t + 1 < T else torch.zeros_like(vf[0])
            print(f"[{i:02d}] 头 {h} 位置 {t:3d}：|fi| {fi[t, h].norm().item():.4f}  |gold| {gold[t, h].norm().item():.3f}  "
                  f"离未来 v 均值 {(fi[t, h] - fut).norm().item():.3f}")


def synth(s: float, T: int):
    """所有 key 相同、所有 q 相同，使每个分数恰好为 s。v 随机。"""
    a = 320.0                                            # 和真实第 0 层 max|k| 同量级
    b = s / (a * SCALE)
    q = torch.zeros(T, H, D, device="cuda"); q[..., 0] = b
    k = torch.zeros(T, HKV, D, device="cuda"); k[..., 0] = a
    v = torch.randn(T, HKV, D, device="cuda")
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def make_wrappers(T: int, sm_scale):
    """独立于 FlashInferBackend 建三个 wrapper，sm_scale=None 即 flashinfer 默认 1/sqrt(D)（修复前的用法）。"""
    import flashinfer
    ws = lambda: torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cu = torch.tensor([0, T], dtype=torch.int32)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=sm_scale)
    c = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws(), kv_layout="NHD")
    c.plan(cu, cu, H, HKV, D, causal=True, **kw)
    nc = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws(), kv_layout="NHD")
    nc.plan(cu, cu, H, HKV, D, causal=False, **kw)
    n_blocks = T // BS
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws(), kv_layout="NHD", use_tensor_cores=True)
    dec.plan(torch.tensor([0, n_blocks], dtype=torch.int32), torch.arange(n_blocks, dtype=torch.int32),
             torch.tensor([BS], dtype=torch.int32), H, HKV, D, BS, **kw)
    return c, nc, dec


@torch.inference_mode()
def sweep():
    print("\n== B. 合成数据：每行分数都是 s，相对误差（vs fp32 SDPA）==")
    print("   旧 = q 原样 + 默认 sm_scale（修复前）；新 = q 预乘 1/sqrt(D) + sm_scale=1.0（现在 Attention 的用法）")
    print("        s    旧:causal  非causal    decode  |  新:causal  非causal    decode")
    T = 64
    n_blocks = T // BS
    old, new = make_wrappers(T, None), make_wrappers(T, 1.0)
    rel = lambda o, g: ((o.float() - g).norm() / g.norm()).item()
    for s in SWEEP:
        q, k, v = synth(s, T)
        qf, kf, vf = q.float(), k.float(), v.float()
        gold_c = sdpa(qf, kf, vf)
        gold_nc = torch.nn.functional.scaled_dot_product_attention(
            qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None], enable_gqa=True)[0].transpose(0, 1)
        kc, vc = k.view(n_blocks, BS, HKV, D).contiguous(), v.view(n_blocks, BS, HKV, D).contiguous()
        row = []
        for (c, nc, dec), qq in ((old, q), (new, q * SCALE)):
            row += [rel(c.run(qq, k, v), gold_c), rel(nc.run(qq, k, v), gold_nc),
                    rel(dec.run(qq[-1:].contiguous(), (kc, vc)), gold_nc[-1:])]   # decode：最后一个 query 看全部 key
        print(f"   {s:7.0f}   " + "  ".join(f"{e:8.2e}" for e in row[:3]) + "  |  "
              + "  ".join(f"{e:8.2e}" for e in row[3:]))


def main():
    ref = load_ref()
    runner = build_runner(backend="flashinfer")
    real_rows(runner, ref)
    sweep()


if __name__ == "__main__":
    main()
