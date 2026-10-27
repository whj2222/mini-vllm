from typing import NamedTuple
from mini_vllm.engine.sequence import Sequence

def slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    assert 0 <= pos < len(block_table) * block_size
    logical_block = pos // block_size
    offset = pos % block_size
    physical_block = block_table[logical_block]
    return offset + physical_block * block_size

def kv_side(seq: Sequence, block_size: int) -> tuple[int, list[int], int]:
    L = len(seq)
    n = (L + block_size - 1) // block_size
    assert len(seq.block_table) >= n
    return L, seq.block_table[:n], L - (n - 1) * block_size

def pad_block_tables(tables: list[list[int]]) -> list[list[int]]:
    width = max(len(t) for t in tables)
    return [t + [-1] * (width - len(t)) for t in tables]

class PrefillInputs(NamedTuple):
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    cu_seqlens: list[int]
    max_seqlen: int
    seq_lens: list[int]
    block_tables: list[list[int]]
    kv_indptr: list[int]
    kv_indices: list[int]
    kv_last_page_len: list[int]
    has_cached: bool

class DecodeInputs(NamedTuple):
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    seq_lens: list[int]
    block_tables: list[list[int]]
    kv_indptr: list[int]
    kv_indices: list[int]
    kv_last_page_len: list[int]

def prepare_prefill(seqs: list[Sequence], block_size: int) -> PrefillInputs:
    input_ids = []
    positions = []
    slot_mapping = []
    cu_seqlens = [0]
    max_seqlen = 0
    seq_lens = []
    block_tables = []
    kv_indices = []
    kv_indptr = [0]
    kv_last_page_len = []
    for seq in seqs:
        assert len(seq.block_table) != 0
        assert seq.num_cached_tokens % block_size == 0
        assert seq.num_cached_tokens < len(seq)
        for i in range(seq.num_cached_tokens, len(seq)):
            token = seq.token_ids[i]
            input_ids.append(token)
            positions.append(i)
            slot_mapping.append(slot_of(seq.block_table, i, block_size))
        cu_seqlens.append(len(seq) - seq.num_cached_tokens + cu_seqlens[-1])
        max_seqlen = max(max_seqlen, len(seq.token_ids) - seq.num_cached_tokens)
        # KV 侧按整段序列算（含命中的前缀），和 decode 相同
        L, blocks, last = kv_side(seq, block_size)
        seq_lens.append(L)
        block_tables.append(seq.block_table)
        kv_indices.extend(blocks)
        kv_indptr.append(kv_indptr[-1] + len(blocks))
        kv_last_page_len.append(last)
    block_tables = pad_block_tables(block_tables)
    has_cached = any(s.num_cached_tokens > 0 for s in seqs)
    return PrefillInputs(input_ids, positions, slot_mapping, cu_seqlens, max_seqlen,
                         seq_lens, block_tables, kv_indptr, kv_indices, kv_last_page_len, has_cached)

def prepare_decode(seqs: list[Sequence], block_size: int) -> DecodeInputs:
    input_ids = []
    positions = []
    slot_mapping = []
    seq_lens = []
    block_tables = []
    kv_indices = []
    kv_indptr = [0]
    kv_last_page_len = []
    for seq in seqs:
        L, blocks, last = kv_side(seq, block_size)
        input_ids.append(seq.last_token)
        positions.append(L - 1)
        slot_mapping.append(slot_of(seq.block_table, L - 1, block_size))
        seq_lens.append(L)
        block_tables.append(seq.block_table)
        kv_indices.extend(blocks)
        kv_indptr.append(kv_indptr[-1] + len(blocks))
        kv_last_page_len.append(last)
    block_tables = pad_block_tables(block_tables)
    return DecodeInputs(input_ids, positions, slot_mapping, seq_lens, block_tables,
                        kv_indptr, kv_indices, kv_last_page_len)

if __name__ == '__main__':
    a = Sequence(0, [10, 11, 12])
    b = Sequence(1, [20, 21, 22, 23, 24])
    a.block_table = [7]
    b.block_table = [2, 9]
    p = prepare_prefill([a, b], block_size=4)
    print(p.input_ids)
    print(p.positions)
    print(p.slot_mapping)
    print(p.cu_seqlens)
    print(p.max_seqlen)

    a = Sequence(0, [10, 11, 12, 13, 14]); a.block_table = [7, 5]
    b = Sequence(1, [20, 21, 22]);         b.block_table = [2]
    d = prepare_decode([a, b], block_size=4)
    print(d.input_ids, d.positions)    # [14, 22] [4, 2]
    print(d.slot_mapping, d.seq_lens)  # [20, 10] [5, 3]
    print(d.block_tables)              # [[7, 5], [2, -1]]
    print(b.block_table)               # [2]