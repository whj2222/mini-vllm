from collections import deque

from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.config import EngineConfig

class Scheduler():
    def __init__(self, engine_cfg, block_manager, eos_token_ids, max_model_len):
        self.engine_cfg = engine_cfg
        self.block_manager = block_manager
        self.eos_token_ids = set(eos_token_ids)
        self.max_model_len = max_model_len
        self.num_preemptions = 0
        self.waiting = deque()
        self.running = deque()

        assert Sequence.block_size == self.block_manager.block_size
        assert self.eos_token_ids

    def add(self, seq: Sequence):
        assert len(seq) + seq.max_tokens <= self.max_model_len
        assert len(seq) <= self.engine_cfg.max_num_batched_tokens
        assert seq.status == SequenceStatus.WAITING and seq.block_table == []
        assert ((len(seq) + seq.max_tokens + self.block_manager.block_size - 1) //
                self.block_manager.block_size) <= len(self.block_manager.blocks)
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self):
        scheduled = []
        num_batched_tokens = 0
        while self.waiting:
            head = self.waiting[0]
            if len(self.running) + 1 > self.engine_cfg.max_num_seqs: break
            if num_batched_tokens + len(head) > self.engine_cfg.max_num_batched_tokens: break
            if not self.block_manager.can_allocate(head): break
            head = self.waiting.popleft()
            self.running.append(head)
            self.block_manager.allocate(head)
            head.status = SequenceStatus.RUNNING
            scheduled.append(head)
            num_batched_tokens += len(head)
        if scheduled:
            return scheduled, True

        while self.running:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    tail = self.running.pop()
                    self.preempt(tail)
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        self.running = deque(scheduled)
        return scheduled, False

    def postprocess(self, seqs: list[Sequence], token_ids):
        assert len(seqs) == len(token_ids)
        finished = []
        for seq, token in zip(seqs, token_ids):
            seq.append_token(token)
            if (not seq.ignore_eos  and seq.last_token in self.eos_token_ids) or seq.num_completion_tokens >= seq.max_tokens:
                finished.append(seq)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return finished

    def preempt(self, seq: Sequence):
        assert seq.status == SequenceStatus.RUNNING
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

if __name__ == "__main__":
    Sequence.block_size = 4
    def run(num_blocks, specs):
        print(specs)
        from mini_vllm.engine.block_manager import BlockManager
        bm = BlockManager(num_blocks, 4)
        scheduler = Scheduler(EngineConfig(), bm, [0], 128)
        seqs = []
        for i, spec in enumerate(specs):
            seq = Sequence(i, [_ for _ in range(spec[0])], 32, True)
            seqs.append(seq)
            scheduler.add(seq)
        step = 0
        while not scheduler.is_finished():
            step+=1
            batch, is_prefill = scheduler.schedule()
            scheduler.postprocess(batch, [1] * len(batch))
            print(step, 'P' if is_prefill else 'D', [(s.seq_id, len(s)) for s in batch], bm.num_free_blocks,
                  scheduler.num_preemptions)
        return seqs, bm, scheduler

    run(10, [(8, 10) for _ in range(3)])