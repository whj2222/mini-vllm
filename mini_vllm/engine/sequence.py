from enum import Enum, auto

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size: int = 16

    def __init__(self, seq_id: int, prompt_token_ids: list[int], max_tokens=16, ignore_eos=False):
        assert max_tokens >= 1
        self.seq_id = seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids = prompt_token_ids.copy()
        self.num_prompt_tokens = len(prompt_token_ids)
        self.block_table = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.num_cached_tokens = 0

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self):
        return (f"Sequence(id={self.seq_id}, status={self.status.name}, "
                f"len={len(self)}, blocks={self.block_table})")

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return len(self) - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return len(self) - (self.num_blocks - 1) * self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

if __name__ == '__main__':
    p = [1, 2, 3]
    seq = Sequence(0, p)
    seq.token_ids.append(99)
    print(p)