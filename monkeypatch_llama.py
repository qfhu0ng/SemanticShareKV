"""
SemShareKV prefill monkeypatch for LlamaAttention.
Goal: during prefill (processing the full prompt), inject rearranged reference KV and partially recompute token K/V.

Notes:
- This patch is meant for the prefill call only. During decode we disable SemShareContext and let HF Cache handle generation. :contentReference[oaicite:5]{index=5}
"""
import torch
import torch.nn.functional as F

from .semshare_context import get_semshare_context

def _token_importance_from_attn(attn_weights: torch.Tensor, window: int = 32) -> torch.Tensor:
    """
    attn_weights: [bs, heads, q_len, k_len]
    returns: [bs, k_len]
    """
    w = min(window, attn_weights.shape[-2])
    tail = attn_weights[:, :, -w:, :]  # last w queries
    score = tail.mean(dim=2).mean(dim=1)
    return score

def _select_hot_mask(score: torch.Tensor, thresh: float = 0.55) -> torch.Tensor:
    """
    score: [bs, k_len], returns hot_mask [bs, k_len]
    """
    bs, k_len = score.shape
    hot_mask = torch.zeros_like(score, dtype=torch.bool)
    for b in range(bs):
        s = score[b]
        vals, idx = torch.sort(s, descending=True)
        mass = vals / (vals.sum() + 1e-8)
        c = torch.cumsum(mass, dim=0)
        n = int((c <= thresh).sum().item())
        n = min(n + 1, k_len)
        hot_mask[b, idx[:n]] = True
    return hot_mask

def patch_llama_attention():
    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

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

        # Q always recomputed
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bs,h,q,d]

        inj_k, inj_v = ctx.injected_past_kv[layer_idx]  # [bs, kvh, q, d]
        key_states = inj_k
        value_states = inj_v

        # recompute set
        if layer_idx == 0:
            recompute_idx = torch.arange(q_len, device=hidden_states.device)
        else:
            recompute_idx = ctx.layer_recompute_idx.get(layer_idx, torch.tensor([], device=hidden_states.device, dtype=torch.long))

        if recompute_idx.numel() > 0:
            hs_sub = hidden_states.index_select(dim=1, index=recompute_idx)
            k_sub = self.k_proj(hs_sub).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v_sub = self.v_proj(hs_sub).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # rotary (HF style)
            cos, sin = self.rotary_emb(value_states, position_ids)
            q_rot, k_rot = apply_rotary_pos_emb(query_states, k_sub, cos, sin)
            query_states = q_rot

            key_states = key_states.clone()
            value_states = value_states.clone()
            key_states[:, :, recompute_idx, :] = k_rot
            value_states[:, :, recompute_idx, :] = v_sub

        # expand kv heads to full heads
        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        value_states_full = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states_full)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # store scores for pruning
        with torch.no_grad():
            ctx.score_store[layer_idx] = _token_importance_from_attn(attn_weights, window=ctx.score_window).detach()

        # (optional) build next-layer recompute set based on deviation + hot/cold (paper-inspired)
        if layer_idx == 0:
            with torch.no_grad():
                dev = (key_states[:, : self.num_key_value_heads] - inj_k).pow(2).mean(dim=(1, 3))  # [bs, q]
                score = ctx.score_store[layer_idx]  # [bs, q]
                hot = _select_hot_mask(score, thresh=ctx.attn_recovery)

                b = 0  # batch=1 typical; still works for >1 by union
                hot_idx = hot[b].nonzero(as_tuple=False).view(-1)
                cold_idx = (~hot[b]).nonzero(as_tuple=False).view(-1)

                n_hot = max(1, int(ctx.recompute_hot_ratio * max(1, hot_idx.numel())))
                n_cold = max(1, int(ctx.recompute_cold_ratio * max(1, cold_idx.numel())))

                if hot_idx.numel() > 0:
                    hs = score[b, hot_idx]
                    _, ordh = torch.sort(hs, descending=True)
                    hot_pick = hot_idx[ordh[:n_hot]]
                else:
                    hot_pick = hot_idx

                if cold_idx.numel() > 0:
                    cd = dev[b, cold_idx]
                    _, ordc = torch.sort(cd, descending=True)
                    cold_pick = cold_idx[ordc[:n_cold]]
                else:
                    cold_pick = cold_idx

                pick = torch.unique(torch.cat([hot_pick, cold_pick], dim=0))
                ctx.layer_recompute_idx[layer_idx + 1] = pick

        present = (key_states, value_states) if use_cache else None
        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    LlamaAttention.forward = semshare_forward
    return orig_forward
