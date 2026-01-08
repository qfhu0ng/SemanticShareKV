"""
SemShareKV monkeypatch for LlamaAttention (Transformers >= 4.5x).

Key points:
- In newer Transformers, upper layers often unpack self_attn outputs as 2 values:
    hidden_states, attn_weights = self.self_attn(...)
  so this patch MUST return exactly 2 values.
- KV cache is typically managed by Cache/DynamicCache in-place, not by returning "present".
  (We avoid returning present to keep compatibility.)
- We reuse ctx.injected_past_kv as the actual KV used in attention, and write back any
  recomputed tokens so the caller can retrieve a complete legacy tuple.
"""
import math
import torch
import torch.nn.functional as F

from ..semshare_context import get_semshare_context


def _token_importance_from_attn(attn_weights: torch.Tensor, window: int = 32) -> torch.Tensor:
    # attn_weights: [bs, heads, q_len, kv_len]
    w = min(window, attn_weights.shape[-2])
    tail = attn_weights[:, :, -w:, :]                  # [bs, heads, w, kv_len]
    return tail.mean(dim=2).mean(dim=1)                # [bs, kv_len]


def patch_llama_attention():
    from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv

    orig_forward = LlamaAttention.forward

    def semshare_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        ctx = get_semshare_context()
        if (not ctx.enabled) or (ctx.injected_past_kv is None):
            return orig_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is None:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.shape

        # Transformers newer LlamaAttention may not expose num_heads attributes,
        # so derive from config.
        num_heads = int(getattr(self, "num_heads", 0) or getattr(self.config, "num_attention_heads", 0))
        head_dim = int(getattr(self, "head_dim", 0) or (getattr(self.config, "hidden_size", 0) // num_heads))
        num_kv_heads = int(
            getattr(self, "num_key_value_heads", 0)
            or getattr(self.config, "num_key_value_heads", 0)
            or num_heads
        )
        num_kv_groups = int(getattr(self, "num_key_value_groups", 0) or (num_heads // num_kv_heads))

        # Project Q (we always compute Q from current hidden_states)
        query_states = self.q_proj(hidden_states)  # [bs, q, hidden]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [bs, h, q, d]

        # Inject KV (already aligned to target length by rearrange_past_kv)
        inj_k, inj_v = ctx.injected_past_kv[layer_idx]  # expect [bs, kvh, q, d]
        key_states = inj_k.to(device=hidden_states.device, dtype=query_states.dtype)
        value_states = inj_v.to(device=hidden_states.device, dtype=query_states.dtype)

        # Optional: recompute a subset of positions (hot tokens) from current hidden_states
        recompute_idx = ctx.layer_recompute_idx.get(
            layer_idx, torch.tensor([], device=hidden_states.device, dtype=torch.long)
        )
        if layer_idx == 0:
            recompute_idx = torch.arange(q_len, device=hidden_states.device, dtype=torch.long)

        if recompute_idx.numel() > 0:
            hs_sub = hidden_states.index_select(dim=1, index=recompute_idx)  # [bs, r, hidden]
            k_sub = self.k_proj(hs_sub).view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)  # [bs, kvh, r, d]
            v_sub = self.v_proj(hs_sub).view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)  # [bs, kvh, r, d]

            # NOTE: Proper RoPE re-application for the subset depends on internal rotary_emb APIs.
            # For robustness (and to avoid version-specific breakage), we skip re-rotating here.
            # You can add RoPE if you lock transformers version & confirm rotary API.

            key_states = key_states.clone()
            value_states = value_states.clone()
            key_states[:, :, recompute_idx, :] = k_sub
            value_states[:, :, recompute_idx, :] = v_sub

        # Write back the actual KV used, so the caller can retrieve it reliably.
        ctx.injected_past_kv[layer_idx] = (key_states, value_states)

        # Expand KV heads to attention heads
        key_states_full = repeat_kv(key_states, num_kv_groups)      # [bs, h, kv_len, d]
        value_states_full = repeat_kv(value_states, num_kv_groups)  # [bs, h, kv_len, d]

        # Attention
        attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_full)  # [bs, h, q, d]

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)

        with torch.no_grad():
            ctx.score_store[layer_idx] = _token_importance_from_attn(attn_weights, window=ctx.score_window).detach()

        # IMPORTANT: return exactly 2 values to match decoder layer unpacking
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None

    LlamaAttention.forward = semshare_forward
    return orig_forward
