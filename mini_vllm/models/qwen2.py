import torch
import torch.nn as nn

from mini_vllm.layers.attention import Attention
from mini_vllm.layers.rotary import RotaryEmbedding
from mini_vllm.layers.mlp import MLP
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.config import ModelConfig

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads,
                                   config.num_key_value_heads, config.head_dim, rotary)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions, hidden_states, attn_metadata):
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states), attn_metadata)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary = RotaryEmbedding(config.head_dim, config.max_model_len, config.rope_theta)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, rotary) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, attn_metadata):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, attn_metadata)
        return self.norm(hidden_states)

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions, attn_metadata):
        hidden = self.model(input_ids, positions, attn_metadata)
        logits = self.lm_head(hidden)
        return logits