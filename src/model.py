# credit to https://github.com/andmev/llama-to-coreml/blob/master/src/model.py

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from .cache import SliceUpdateKeyValueCache

class KvCacheStateLlamaForCausalLM(torch.nn.Module):
    """Model wrapper to swap cache implementation and register as buffers."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 1,
        context_size: int = 4096,
        eos_token_id: int = 2
    ) -> None:
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        config: LlamaConfig = self.model.config
        config.eos_token_id = eos_token_id
        self.kv_cache_shape: tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            context_size,
            config.hidden_size // config.num_attention_heads,
        )
        # Register KV cache buffers to be recognized as Core ML states
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k)
        self.register_buffer("valueCache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        if causal_mask is None:
            self.kv_cache.past_seen_tokens = input_ids.shape[-1]
        else:
            self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits