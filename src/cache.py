# credit to https://github.com/andmev/llama-to-coreml/blob/master/src/cache.py

from typing import Any, Optional, Sequence
import torch
from transformers.cache_utils import Cache

class SliceUpdateKeyValueCache(Cache):
    """Helper class for in-place slice updating key/value caches."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).
        Return slice of key / value cache tensors from [0, end)."""
        position = cache_kwargs.get("cache_position", None)
        assert position is not None, "cache_position required to update cache."
        begin, end = self.past_seen_tokens, self.past_seen_tokens + position.shape[-1]
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_state = self.k[layer_idx, :, :, :end, :]
        v_state = self.v[layer_idx, :, :, :end, :]
        return k_state, v_state

    def get_seq_length(self, _: int = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens