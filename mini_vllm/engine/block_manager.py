from collections import OrderedDict
from mini_vllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return (f"Block(id={self.block_id}, ref_count={self.ref_count})")

    def update(self, h, token_ids):
        self.hash = h
        self.token_ids = tuple(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: OrderedDict = OrderedDict.fromkeys(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def compute_hash(self, token_ids, prefix_hash):
        return hash((tuple(token_ids), prefix_hash))

    def _match_prefix(self, seq):
        """
        缓存命中五个条件：
        1. 前面的块全部命中
        2. 这一块是满的
        3. hash在表里
        4. 内容相同

        调用方：can_allocate, allocate
        """
        hit_blocks = []
        h = -1
        for i in range((len(seq) - 1) // self.block_size):
            block_token_ids = seq.block_token_ids(i)
            if len(block_token_ids) != seq.block_size:
                break
            h = self.compute_hash(block_token_ids, h)
            if h not in self.hash_to_block_id:
                break
            block_id = self.hash_to_block_id[h]
            if self.blocks[block_id].token_ids != tuple(block_token_ids):
                break
            hit_blocks.append(self.hash_to_block_id[h])
        return hit_blocks


    def _allocate_new_block(self) -> Block:
        block_id, _ = self.free_block_ids.popitem(last=False)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _allocated_cached_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        if block.ref_count == 0:
            del self.free_block_ids[block_id]
            self.used_block_ids.add(block_id)
        elif block.ref_count >= 1:
            pass
        block.ref_count += 1
        return block


    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids[block_id] = None

    def can_allocate(self, seq: Sequence) -> bool:
        hit = self._match_prefix(seq)
        num_hit_free = sum(1 for block_id in hit if self.blocks[block_id].ref_count == 0)
        return (seq.num_blocks - len(hit)) + num_hit_free <= self.num_free_blocks

    def allocate(self, seq: Sequence) -> None:
        assert seq.block_table == []
        assert self.can_allocate(seq)
        hit = self._match_prefix(seq)
        for block_id in hit:
            seq.block_table.append(self._allocated_cached_block(block_id).block_id)
        for i in range(len(hit), seq.num_blocks):
            seq.block_table.append(self._allocate_new_block().block_id)
            block_ids = seq.block_token_ids(i)
            self.blocks[seq.block_table[i]].token_ids = tuple(block_ids)
            if len(block_ids) == self.block_size:
                h = self.compute_hash(block_ids, self.blocks[seq.block_table[i-1]].hash)
                self.hash_to_block_id[h] = seq.block_table[i]
            else:
                h = -1
            self.blocks[seq.block_table[i]].hash = h
        seq.num_cached_tokens = len(hit) * self.block_size


    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table = []

    def can_append(self, seq: Sequence) -> bool:
        need_new_token = seq.last_block_num_tokens
        if need_new_token == 1 and self.num_free_blocks == 0:
            return False
        return True

    def may_append(self, seq: Sequence) -> None:
        if seq.last_block_num_tokens == 1:
            assert len(self.free_block_ids) != 0
            block_id = self._allocate_new_block().block_id
            seq.block_table.append(block_id)
        if seq.last_block_num_tokens == seq.block_size:
            block_id = seq.block_table[-1]
            prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) >= 2 else -1
            h = self.compute_hash(seq.block_token_ids(seq.num_blocks - 1), prefix)
            if h not in self.hash_to_block_id:
                self.hash_to_block_id[h] = block_id
            self.blocks[block_id].update(h, seq.block_token_ids(seq.num_blocks - 1))
        assert len(seq.block_table) == seq.num_blocks


if __name__ == "__main__":
    Sequence.block_size = 4
    bm = BlockManager(num_blocks=10, block_size=4)

    # prefill + 释放
    a = Sequence(0, list(range(6)))
    b = Sequence(1, list(range(9)))
    bm.allocate(a); bm.allocate(b)
    print(a.block_table, b.block_table, bm.num_free_blocks)   # [0, 1] [2, 3, 4] 5
    bm.deallocate(a)
    print(a.block_table, bm.num_free_blocks)                  # [] 7
    bm.allocate(a)
    print(a.block_table)                                      # [5, 6]
    bm.deallocate(a); bm.deallocate(b)
    print(bm.num_free_blocks, bm.used_block_ids)              # 10 set()

    # decode
    s = Sequence(2, list(range(3)))
    bm.allocate(s)
    for t in range(6):
        s.append_token(100 + t)
        assert bm.can_append(s)
        bm.may_append(s)
        print(len(s), s.block_table, bm.num_free_blocks)
