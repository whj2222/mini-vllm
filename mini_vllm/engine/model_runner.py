import torch
from torch import Tensor

from mini_vllm.engine.sequence import Sequence
from mini_vllm.models.qwen2 import Qwen2ForCausalLM
from mini_vllm.config import load_model_config, EngineConfig
from mini_vllm.utils.loader import load_weights
from mini_vllm.engine import input_prep
from mini_vllm.layers.attn_metadata import AttentionMetadata


class ModelRunner:
    def __init__(self, model_path: str, engine_config: EngineConfig):
        self.engine_cfg = engine_config
        self.model_path = model_path

        self.model_cfg = load_model_config(model_path, self.engine_cfg.max_model_len)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                self.model = Qwen2ForCausalLM(self.model_cfg)
        finally:
            torch.set_default_dtype(old_dtype)

        load_weights(self.model, model_path)

        params = self.model.parameters()
        param = next(params)
        self.dtype = param.dtype
        self.device = param.device
        if self.engine_cfg.attn_backend == "flashinfer":
            from mini_vllm.layers.flashinfer_backend import FlashInferBackend
            self.attn_backend = FlashInferBackend(self.model_cfg.num_attention_heads, self.model_cfg.num_key_value_heads,
                                                  self.model_cfg.head_dim, self.engine_cfg.block_size, dtype=self.dtype, device=self.device)
        else:
            self.attn_backend = None
        if self.engine_cfg.num_gpu_blocks == None:
            self.engine_cfg.num_gpu_blocks = self._profile_num_blocks()

        self._allocate_kv_cache(self.engine_cfg.num_gpu_blocks)

        self.graph_bs = [1, 2, 4] + list(range(8, 256 + 1, 8))
        self.graph_bs = [b for b in self.graph_bs if b <= self.engine_cfg.max_graph_bs]
        self.bs_to_graph_bs = [0] * (self.graph_bs[-1] + 1)
        p = 0
        for n in range(1, len(self.bs_to_graph_bs)):
            if self.graph_bs[p] < n:
                p += 1
            self.bs_to_graph_bs[n] = self.graph_bs[p]

        self.use_graph = (not self.engine_cfg.enforce_eager and self.engine_cfg.attn_backend == "flashinfer")
        if self.use_graph:
            self.capture_graphs()

    def capture_graphs(self):
        pass

    def _run_decode_graphs(self, seqs: list[Sequence]) -> list[int]:
        raise NotImplementedError

    def _kv_byte_per_block(self) -> int:
        return (self.engine_cfg.block_size * self.model_cfg.num_hidden_layers * 2 *
                self.model_cfg.num_key_value_heads * self.model_cfg.head_dim * self.dtype.itemsize)

    @torch.inference_mode()
    def _profile_num_blocks(self) -> int:
        cfg = self.model_cfg
        ecfg = self.engine_cfg
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


        T = ecfg.max_num_batched_tokens
        max_len = cfg.max_model_len
        cu = list(range(0, T, max_len)) + [T]
        input_ids = torch.zeros(T, dtype=int, device=self.device)
        positions = torch.arange(T, device=self.device) % cfg.max_model_len
        if self.attn_backend is not None:
            wrapper = self.attn_backend.plan_prefill(torch.tensor(cu, dtype=torch.int32))
        else:
            wrapper = None
        md = AttentionMetadata(is_prefill=True, slot_mapping=torch.zeros(T, dtype=torch.int32, device=self.device),
                               cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=self.device),
                                max_seqlen=4096, wrapper=wrapper)
        before = torch.cuda.memory_allocated()
        hidden = self.model.model(input_ids, positions, md)
        logits = self.model.lm_head(hidden[-1:])
        torch.cuda.synchronize()

        activation = torch.cuda.max_memory_allocated() - before

        del input_ids, positions, hidden, logits
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free

        avaliable = total * ecfg.gpu_memory_utilization - used - activation
        num_blocks = int(avaliable // self._kv_byte_per_block())
        if num_blocks < 0 or num_blocks * self.engine_cfg.block_size < self.engine_cfg.max_model_len:
            raise RuntimeError(
                f"KV 池放不下一条最长序列：num_blocks={num_blocks}，"
                f"需要 ≥ {self.model_cfg.max_model_len / self.engine_cfg.block_size:.0f} 个 block。"
                f"total={total / 2 ** 30:.2f}G used={used / 2 ** 30:.2f}G activation={activation / 2 ** 30:.2f}G")
        return num_blocks


    def _allocate_kv_cache(self, num_blocks: int) -> None:
        # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(2, self.model_cfg.num_hidden_layers, num_blocks,
                   self.engine_cfg.block_size, self.model_cfg.num_key_value_heads,
                   self.model_cfg.head_dim,dtype=self.dtype, device=self.device)
        for i in range(self.model_cfg.num_hidden_layers):
            k_cache = self.kv_cache[0, i, :, :, :, :]
            v_cache = self.kv_cache[1, i, :, :, :, :]
            self.model.model.layers[i].self_attn.k_cache = k_cache
            self.model.model.layers[i].self_attn.v_cache = v_cache

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor, AttentionMetadata]:
        prefill_input = input_prep.prepare_prefill(seqs, self.engine_cfg.block_size)
        input_ids = torch.tensor(prefill_input.input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(prefill_input.positions, dtype=torch.long, device=self.device)

        is_prefill = True
        slot_mapping = torch.tensor(prefill_input.slot_mapping, dtype=torch.int32, device=self.device)
        cu_seqlens = torch.tensor(prefill_input.cu_seqlens, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor(prefill_input.seq_lens, dtype=torch.int32, device=self.device)
        block_tables = torch.tensor(prefill_input.block_tables, dtype=torch.int32, device=self.device)
        max_seqlen = prefill_input.max_seqlen
        if self.attn_backend is not None:
            if prefill_input.has_cached:
                wrapper = self.attn_backend.plan_prefill_paged(torch.tensor(prefill_input.cu_seqlens, dtype=torch.int32,),
                                                               torch.tensor(prefill_input.kv_indptr, dtype=torch.int32,),
                                                               torch.tensor(prefill_input.kv_indices, dtype=torch.int32,),
                                                               torch.tensor(prefill_input.kv_last_page_len, dtype=torch.int32,))
            else:
                wrapper = self.attn_backend.plan_prefill(torch.tensor(prefill_input.cu_seqlens, dtype=torch.int32))
        else:
            wrapper = None
        attention_metadata = AttentionMetadata(is_prefill=is_prefill, slot_mapping=slot_mapping,
                                               cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, wrapper=wrapper,
                                               has_cached=prefill_input.has_cached, seq_lens=seq_lens,
                                               block_tables=block_tables)
        return input_ids, positions, attention_metadata

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor, AttentionMetadata]:
        decode_input = input_prep.prepare_decode(seqs, self.engine_cfg.block_size)
        input_ids = torch.tensor(decode_input.input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(decode_input.positions, dtype=torch.long, device=self.device)

        is_prefill = False
        slot_mapping = torch.tensor(decode_input.slot_mapping, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor(decode_input.seq_lens, dtype=torch.int32, device=self.device)
        block_tables = torch.tensor(decode_input.block_tables, dtype=torch.int32, device=self.device)
        if self.attn_backend is not None:
            wrapper = self.attn_backend.plan_decode(torch.tensor(decode_input.kv_indptr, dtype=torch.int32),
                                                    torch.tensor(decode_input.kv_indices, dtype=torch.int32),
                                                    torch.tensor(decode_input.kv_last_page_len, dtype=torch.int32))
        else:
            wrapper = None
        attention_metadata = AttentionMetadata(is_prefill=is_prefill, slot_mapping=slot_mapping,
                                               seq_lens=seq_lens, block_tables=block_tables, wrapper=wrapper)
        return input_ids, positions, attention_metadata

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not is_prefill and self.use_graph and len(seqs) <= self.graph_bs[-1]:
            return self._run_decode_graphs(seqs)
        if is_prefill:
            input_ids, positions, attention_metadata = self.prepare_prefill(seqs)
        else:
            input_ids, positions, attention_metadata = self.prepare_decode(seqs)
        hidden = self.model.model(input_ids, positions, attention_metadata)
        if is_prefill:
            last_idx = (attention_metadata.cu_seqlens[1:] - 1).long()
            hidden = hidden[last_idx, :]
        logits = self.model.lm_head(hidden)

        return logits.argmax(dim=-1).tolist()


if __name__ == "__main__":
    import json
    import time
    from mini_vllm.engine.block_manager import BlockManager

    MODEL_PATH = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
    REF = json.load(open("scripts/ref.json"))          # 在项目根目录运行
    PROMPTS = [r["input_ids"] for r in REF]
    NEWS = [r["new_ids"] for r in REF]

    def generate(runner, prompts, news):
        """teacher forcing：同批 prefill，之后每步喂 ref 里的 token 做 decode。
        返回 preds，preds[i][k] 是第 i 条对 news[i][k] 的预测。"""
        bs = runner.engine_cfg.block_size
        Sequence.block_size = bs
        bm = BlockManager(runner.engine_cfg.num_gpu_blocks, bs)
        seqs = [Sequence(i, p) for i, p in enumerate(prompts)]
        for s in seqs:
            assert bm.can_allocate(s)
            bm.allocate(s)

        preds = [[t] for t in runner.run(seqs, is_prefill=True)]
        for k in range(max(len(n) for n in news) - 1):
            # 长度不同的 seq 陆续退出，后面的步数里 batch 越来越小
            active = [s for s in seqs if k + 1 < len(news[s.seq_id])]
            for s in active:
                s.append_token(news[s.seq_id][k])
                assert bm.can_append(s)
                bm.may_append(s)
            for s, t in zip(active, runner.run(active, is_prefill=False)):
                preds[s.seq_id].append(t)

        for s in seqs:
            bm.deallocate(s)
        assert bm.num_free_blocks == runner.engine_cfg.num_gpu_blocks
        return preds

    def match(preds, news):
        ok = sum(p == t for P, N in zip(preds, news) for p, t in zip(P, N))
        return ok, sum(len(n) for n in news)

    def check(runner, tag):
        # 1. 单条
        ok, total = match(generate(runner, PROMPTS[:1], NEWS[:1]), NEWS[:1])
        print(f"[{tag}] 单条      {ok}/{total}")

        # 2. 20 条同批
        batch = generate(runner, PROMPTS, NEWS)
        ok, total = match(batch, NEWS)
        print(f"[{tag}] 20 条同批 {ok}/{total} = {ok / total:.2%}")
        assert ok / total >= 0.98, "一致率低于 98%"

        # 3. 同批 vs 逐条单跑：串台的话这里会大面积不同
        single = [generate(runner, [PROMPTS[i]], [NEWS[i]])[0] for i in range(len(REF))]
        diff = [(i, k) for i, (A, B) in enumerate(zip(batch, single))
                for k, (a, b) in enumerate(zip(A, B)) if a != b]
        print(f"[{tag}] 同批 vs 单跑 不同 {len(diff)} 处 {diff[:10]}")

    # ---- profile + block_size=16 ----
    t0 = time.time()
    runner = ModelRunner(MODEL_PATH, EngineConfig())
    n = runner.engine_cfg.num_gpu_blocks
    free, total = torch.cuda.mem_get_info()
    print(f"num_blocks={n}  tokens={n * runner.engine_cfg.block_size}  "
          f"KV={runner.kv_cache.numel() * runner.kv_cache.element_size() / 2**30:.2f} GB  "
          f"used={(total - free) / total:.3f}  耗时 {time.time() - t0:.1f}s")
    check(runner, "bs=16")

    # ---- block_size=1：每个 token 一块，slot 算错最容易暴露 ----
    del runner
    torch.cuda.empty_cache()
    runner = ModelRunner(MODEL_PATH, EngineConfig(block_size=1, num_gpu_blocks=100_000))
    check(runner, "bs=1")
    print("全部通过")