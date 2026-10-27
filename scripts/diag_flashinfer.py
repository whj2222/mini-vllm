"""flashinfer 数值诊断：test_ref_alignment[flashinfer] 里第 6 条那几处 gap 0.75，是数值差异还是 bug。

ref.json 是 HF bf16 + SDPA 在 A4000 上生成的，torch 后端也用 SDPA，天然更像它；
所以拿 ref.json 当真值对 flashinfer 不公平。这里用两种"真值"重新判：

1. 算子级：同一组真实 q/k/v（第 6、16 条 prefill 时每层的输入），以 fp32 attention 为真值，
   比 bf16 SDPA 和 flashinfer 各自的误差。两者同量级 → 数值差异；某层大一个数量级 → bug。
2. 端到端：HF fp32 跑 20 条（teacher forcing），它的 top-1 当真值，
   看 torch / flashinfer 两个后端谁更接近；再看 flashinfer 那几处大 gap 在 fp32 下本来差多少。

只读，不改任何代码。运行（项目根目录）：python -m scripts.diag_flashinfer
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mini_vllm.config import resolve_model_path
from mini_vllm.layers.attention import Attention
from mini_vllm.tests.runner_utils import MAX_GAP, MODEL_PATH, build_runner, load_ref, prefill_logits

OP_SEQS = [6, 16]               # 失败集中的两条
torch.backends.cuda.matmul.allow_tf32 = False   # fp32 真值不能偷偷用 TF32
torch.backends.cudnn.allow_tf32 = False


def sdpa(q, k, v):
    """[T, H, D] 的 causal GQA attention，dtype 跟输入走。"""
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                       v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
    return o[0].transpose(0, 1)


@torch.inference_mode()
def capture_qkv(runner, ids):
    """走 torch 后端 prefill 一遍，记下每层 _prefill_attn 收到的 q/k/v（不改源码，只临时包一层）。"""
    rec, orig, fi = [], Attention._prefill_attn, runner.attn_backend

    def spy(self, q, k, v, md):
        rec.append((q.clone(), k.clone(), v.clone()))
        return orig(self, q, k, v, md)

    Attention._prefill_attn, runner.attn_backend = spy, None
    try:
        prefill_logits(runner, ids, [])
    finally:
        Attention._prefill_attn, runner.attn_backend = orig, fi
    return rec


@torch.inference_mode()
def op_level(runner, ref):
    print("== 1. 算子级：相对误差 ||o - o_fp32|| / ||o_fp32||，每层同一组 q/k/v ==")
    for i in OP_SEQS:
        ids = ref[i]["input_ids"] + ref[i]["new_ids"]
        wrapper = runner.attn_backend.plan_prefill(torch.tensor([0, len(ids)], dtype=torch.int32))
        e_sdpa, e_fi = [], []
        for q, k, v in capture_qkv(runner, ids):
            gold = sdpa(q.float(), k.float(), v.float())
            n = gold.norm()
            e_sdpa.append(((sdpa(q, k, v).float() - gold).norm() / n).item())
            e_fi.append(((wrapper.run(q * q.shape[-1] ** -0.5, k, v).float() - gold).norm() / n).item())
        ratio = [f / s for f, s in zip(e_fi, e_sdpa)]
        worst = max(range(len(ratio)), key=ratio.__getitem__)
        print(f"[{i:02d}] T={len(ids)}  SDPA 平均 {sum(e_sdpa) / len(e_sdpa):.2e}  "
              f"flashinfer 平均 {sum(e_fi) / len(e_fi):.2e}  "
              f"flashinfer/SDPA 中位 {sorted(ratio)[len(ratio) // 2]:.2f}x  "
              f"最大 {ratio[worst]:.2f}x（第 {worst} 层）")
        print("     每层 flashinfer/SDPA：" + " ".join(f"{r:.1f}" for r in ratio))


@torch.inference_mode()
def end_to_end(runner, ref):
    print("\n== 2. 端到端：以 HF fp32 的 top-1 为真值 ==")
    hf = AutoModelForCausalLM.from_pretrained(resolve_model_path(MODEL_PATH), dtype=torch.float32,
                                              attn_implementation="eager").cuda().eval()
    gold = []                                   # 每条 [N, V] 的 fp32 logits
    for r in ref:
        T, N = len(r["input_ids"]), len(r["new_ids"])
        ids = torch.tensor([r["input_ids"] + r["new_ids"]], device="cuda")
        gold.append(hf(ids).logits[0, T - 1: T - 1 + N].float())
    del hf
    torch.cuda.empty_cache()

    fi = runner.attn_backend
    total = sum(len(r["new_ids"]) for r in ref)
    gold_top = [g.argmax(-1) for g in gold]
    ref_vs_gold = sum((gt.cpu() == torch.tensor(r["new_ids"])).sum().item() for gt, r in zip(gold_top, ref))
    print(f"ref.json（HF bf16）vs fp32：{ref_vs_gold}/{total}（{ref_vs_gold / total:.2%}）")

    for name, backend in (("torch", None), ("flashinfer", fi)):
        runner.attn_backend = backend
        agree, big = 0, []
        for i, (r, g, gt) in enumerate(zip(ref, gold, gold_top)):
            P = prefill_logits(runner, r["input_ids"], r["new_ids"])
            mine = P.argmax(-1)
            agree += (mine == gt).sum().item()
            for k, (m, t) in enumerate(zip(mine.tolist(), r["new_ids"])):
                gap = (P[k, m] - P[k, t]).item()
                if m != t and gap > MAX_GAP:
                    # fp32 下参考 token 比我的 token 高多少：小 → 这个位置本来就险，bf16 谁都可能翻
                    big.append((i, k, m, t, gap, (g[k, t] - g[k, m]).item(), g[k].max().item()))
        print(f"[{name:10s}] vs fp32 top-1 {agree}/{total}（{agree / total:.2%}）")
        for i, k, m, t, gap, margin, top in big:
            print(f"    第 {i} 条 k={k}：我 {m} / ref {t}，bf16 gap {gap:.3f}，"
                  f"fp32 下 ref 领先 {margin:+.3f}（fp32 top-1 logit {top:.2f}）")
    runner.attn_backend = fi


def main():
    ref = load_ref()
    runner = build_runner(backend="flashinfer")
    op_level(runner, ref)
    end_to_end(runner, ref)


if __name__ == "__main__":
    main()
