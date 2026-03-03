
import dataclasses
import logging
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from typing import Any

from openpi.models import model as _model
from openpi.models import pi0_fast as _pi0_fast
from openpi.models import pi0 as _pi0  # For posemb_sincos
import openpi.models.gemma_fast as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

@dataclasses.dataclass(frozen=True)
class Pi0FASTHybridConfig(_pi0_fast.Pi0FASTConfig):
    # Hybrid specific config
    track_dim: int = 128  # 64 points * 2 coords
    track_horizon: int = 16
    
    # Weights for loss
    action_loss_weight: float = 1.0
    track_loss_weight: float = 1.0
    
    # Increase default max_token_len to handle query points in prompt
    max_token_len: int = 512

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST  # Reuse type or define new one if needed

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTHybrid":
        return Pi0FASTHybrid(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, dict]:
        # Get base specs
        obs_spec, _ = super().inputs_spec(batch_size=batch_size)
        
        # Define hybrid actions spec
        action_spec = {
            "token_actions": jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32),
            "track_actions": jax.ShapeDtypeStruct([batch_size, self.track_horizon, self.track_dim], jnp.float32),
        }
        return obs_spec, action_spec

    def get_freeze_filter(self) -> Any:
        # Custom freeze filter to support "track_head_only" or "action_head_only"
        # This will be overridden by the training script usually, but good to have defaults
        return super().get_freeze_filter()


class Pi0FASTHybrid(_pi0_fast.Pi0FAST):
    def __init__(self, config: Pi0FASTHybridConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.config = config
        
        # Track Head Components (Diffusion)
        # Using action_expert_config width (usually 1024 or similar depending on Gemma model)
        # We need to access the width from the internal gemma config
        # The base class initializes self.PaliGemma.llm
        # We can assume it uses the standard dimensions.
        
        # Get width from the config passed to gemma
        # We can re-fetch config to get width
        action_expert_config = _gemma.get_config(config.paligemma_variant)
        embed_dim = action_expert_config["width"] # check key
        
        # Track projections
        self.track_dim = config.track_dim
        self.track_horizon = config.track_horizon
        
        # Input projection for tracks (noisy tracks -> embedding)
        self.track_in_proj = nnx.Linear(config.track_dim, embed_dim, rngs=rngs)
        
        # Time MLP
        self.track_time_mlp_in = nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
        self.track_time_mlp_out = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
        # Output projection (embedding -> velocity)
        self.track_out_proj = nnx.Linear(embed_dim, config.track_dim, rngs=rngs)

    def embed_track_suffix(
        self, 
        conditioning_embedding: at.Float[at.Array, "b emb"], 
        noisy_tracks: at.Float[at.Array, "b h d"], 
        timestep: at.Float[at.Array, " b"]
    ) -> at.Float[at.Array, "b s emb"]:
        
        # Conditioning embedding comes from the transformer (last prompt token)
        # It serves as the context for the track diffusion
        
        # Embed timestep
        time_emb = _pi0.posemb_sincos(timestep, self.track_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        # Embed noisy tracks
        track_tokens = self.track_in_proj(noisy_tracks)
        
        # Expand time embedding
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.track_horizon)
        
        # Mix track + time
        track_time_tokens = jnp.concatenate([track_tokens, time_tokens], axis=-1)
        track_time_tokens = self.track_time_mlp_in(track_time_tokens)
        track_time_tokens = nnx.swish(track_time_tokens)
        track_time_tokens = self.track_time_mlp_out(track_time_tokens)
        
        # Combine with conditioning
        # Here we need to decide how to combine.
        # In Pi0, the suffix attends to the prefix.
        # But here we are just running a small head on top of the embedding?
        # Wait, Pi0Tracks runs the transformer on the suffix.
        # Pi0FASTHybrid already ran the transformer on the prompt.
        # If we want to use diffusion *head*, usually we use a small MLP or Transformer Decoder.
        # But Pi0 uses the *same* transformer for diffusion.
        
        # If we want to use the SAME transformer for track diffusion, we would need to append track tokens to the prompt
        # and run the transformer again (or in the same pass).
        # But Pi0FAST is autoregressive for actions.
        # Can we do: [Prompt] -> [Action Tokens] (AR)
        #            [Prompt] -> [Track Tokens] (Diffusion)?
        
        # If we re-use the transformer, we need to construct a batch that has both targets?
        # Or doing 2 passes?
        
        # The user said: "freezes the action head and the backbone and just trains the track head only"
        # This implies the track head might be a separate small network?
        # OR just the adapter layers (projections) around the FROZEN backbone.
        
        # If using frozen backbone, we typically:
        # 1. Run backbone on prompt -> get embeddings.
        # 2. Append track tokens.
        # 3. Run backbone on track tokens (attending to prompt).
        # 4. Predict output.
        
        # So yes, we need to run the backbone on the track suffix.
        return track_time_tokens

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: dict,
        *,
        train: bool = False,
        return_metrics: bool = False,
    ) -> at.Float[at.Array, ""]:
        # Unpack actions
        # If passed as dict
        if isinstance(actions, dict):
            target_tracks = actions.get("track_actions") # (B, H, D)
        else:
            # Fallback or error
            target_tracks = None
            logger.warning("compute_loss received non-dict actions in Hybrid model")

        # 1. Preprocess Observation
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # 2. Embed Inputs (Images + Prompt + Actions) -> Prefix
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        
        # 3. Prepare Track Suffix (if tracks provided)
        # We perform a single forward pass with [Prefix, Suffix]
        # Prefix handles Autoregressive Action Prediction
        # Suffix handles Track Diffusion
        
        # Determine shapes
        batch_size, prefix_len, embed_dim = input_token_embeddings.shape
        
        if target_tracks is not None:
            # Prepare diffusion noise
            track_rng, time_rng = jax.random.split(rng, 2)
            noise = jax.random.normal(track_rng, target_tracks.shape)
            # Sample time
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            # Noisy tracks
            x_t = time_expanded * noise + (1 - time_expanded) * target_tracks
            u_t = noise - target_tracks
            
            # Embed Track Suffix
            track_suffix_tokens = self.embed_track_suffix(None, x_t, time)
            suffix_len = self.track_horizon
            
            # Combine Embeddings
            combined_embeddings = jnp.concatenate([input_token_embeddings, track_suffix_tokens], axis=1)
            
            # Construct Combined Attention Mask
            # Prefix Mask: Standard Pi0FAST mask (from make_attn_mask)
            prefix_attn_mask = _pi0_fast.make_attn_mask(input_mask, ar_mask) # (B, P, P)
            
            # Suffix Mask:
            # Suffix -> Prefix: Attend only to non-action parts of prompt
            # Action tokens in prompt are where token_loss_mask is True
            action_mask = observation.token_loss_mask # (B, P_tokens)
            
            # Pad action_mask to match input_mask (prepend zeros for images)
            # input_mask length = 768 (256*2 images + 256 tokens)
            # action_mask length = 256
            # We need to prepend (768-256) zeros.
            num_image_tokens = input_mask.shape[1] - action_mask.shape[1]
            if num_image_tokens > 0:
                padding = jnp.zeros((batch_size, num_image_tokens), dtype=bool)
                full_action_mask = jnp.concatenate([padding, action_mask], axis=1)
            else:
                full_action_mask = action_mask
                
            prompt_mask = jnp.logical_and(input_mask, jnp.logical_not(full_action_mask)) # (B, P)
            
            suffix_to_prefix_mask = einops.repeat(prompt_mask, "b p -> b s p", s=suffix_len) # (B, S, P)
            
            # Suffix -> Suffix: Full attention
            suffix_to_suffix_mask = jnp.ones((batch_size, suffix_len, suffix_len), dtype=bool) # (B, S, S)
            
            # Prefix -> Suffix: False (Prefix cannot see tracks)
            prefix_to_suffix_mask = jnp.zeros((batch_size, prefix_len, suffix_len), dtype=bool) # (B, P, S)
            
            # Assemble Full Mask
            # [P->P, P->S]
            # [S->P, S->S]
            top = jnp.concatenate([prefix_attn_mask, prefix_to_suffix_mask], axis=2)
            bottom = jnp.concatenate([suffix_to_prefix_mask, suffix_to_suffix_mask], axis=2)
            full_attn_mask = jnp.concatenate([top, bottom], axis=1) # (B, P+S, P+S)
            
            # Expand mask for heads (B, 1, T, T) required by Gemma
            # But make_attn_mask returns (B, T, T).
            # Gemma expects (B, 1, T, T) or (B, H, T, T).
            # pi0_fast.py calls llm with (B, T, T) mask?
            # Let's check pi0_fast.py again.
            # mask=attn_mask[:, :-1, :-1]
            # gemma_fast.py: if mask.ndim == 3: mask = mask[:, None, :, :]
            # So (B, T, T) is fine.
            
            # Construct Positions
            # Prefix positions
            prefix_positions = jnp.cumsum(input_mask, axis=1) - 1 # (B, P)
            # Suffix positions: continue from max prefix position?
            # Or just separate positions?
            # Usually we want them to be distinct.
            # Let's use max(prefix_positions) + 1 + arange(S)
            max_prefix_pos = jnp.max(prefix_positions, axis=1, keepdims=True)
            suffix_positions = max_prefix_pos + 1 + jnp.arange(suffix_len)[None, :] # (B, S)
            full_positions = jnp.concatenate([prefix_positions, suffix_positions], axis=1)
            
        else:
            # No tracks, just prefix
            combined_embeddings = input_token_embeddings
            full_attn_mask = _pi0_fast.make_attn_mask(input_mask, ar_mask)
            full_positions = jnp.cumsum(input_mask, axis=1) - 1
            track_loss = 0.0

        # 4. Forward Pass (Single Pass)
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=combined_embeddings,
            mask=full_attn_mask,
            positions=full_positions,
            return_prelogits=True,
        )
        
        # 5. Compute Action Loss (on Prefix part)
        # We need pre_logits for [:-1] of prefix to predict [1:] of prefix.
        prefix_pre_logits = pre_logits[:, :prefix_len]
        
        # Same logic as Pi0FAST
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )
        
        # Use prefix_pre_logits[:, :-1] to predict targets
        # We need to run decode (head) on these
        action_logits, _ = self.PaliGemma.llm(
            pre_logits=prefix_pre_logits[:, :-1],
        )
        # Wait, targets shape is (B, P-1, Vocab)
        # action_logits shape should be (B, P-1, Vocab)
        
        # We only need to compute logits for the targets (to save memory)
        # Same optimization as Pi0FAST:
        # logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1] :])
        # Here we need to be careful with slicing because pre_logits might contain suffix.
        
        # We want the last P-1 tokens of prefix.
        # But wait, targets correspond to `tokenized_prompt[:, 1:]`.
        # `tokenized_prompt` is length P.
        # So we have P-1 targets.
        # We predict them from `input_token_embeddings[:, :-1]`.
        # So we use `prefix_pre_logits[:, :-1]`.
        
        # Optimization: Only compute logits for tokens where loss_mask is True?
        # Pi0FAST optimization: `pre_logits=pre_logits[:, -targets.shape[1] :]`
        # This implies it only computes logits for the *end* of the sequence?
        # Ah, in Pi0FAST, `input_token_embeddings` includes image tokens + prompt tokens.
        # `targets` is only `tokenized_prompt`.
        # `tokenized_prompt` is much shorter than `input_token_embeddings` (images are ~256 tokens).
        # So `targets.shape[1]` is small.
        # `prefix_pre_logits` has length Image+Prompt.
        # We want the last `targets.shape[1]` vectors from `prefix_pre_logits`?
        # Yes, because `tokenized_prompt` is at the end of the input.
        
        # So:
        num_targets = targets.shape[1]
        action_logits_input = prefix_pre_logits[:, -num_targets:]
        # Wait, prefix_pre_logits includes the last token (which predicts nothing/next).
        # We want to predict `tokenized_prompt[1:]`.
        # The first token of `tokenized_prompt` predicts the second.
        # `prefix_pre_logits` corresponds to inputs.
        # So we want the slice corresponding to `tokenized_prompt[:-1]`.
        # `tokenized_prompt` starts after images.
        # So `prefix_pre_logits` [Images:] corresponds to `tokenized_prompt`.
        # We want `prefix_pre_logits` [Images : -1].
        # Length is `len(tokenized_prompt) - 1`.
        # Which is exactly `num_targets`.
        
        # But Pi0FAST does:
        # `pre_logits, _, _ = self.PaliGemma.llm(embedded_prefix=input_token_embeddings[:, :-1], ...)`
        # It runs forward on `[:-1]`.
        # And then takes `pre_logits[:, -targets.shape[1] :]`.
        
        # In my code, I ran forward on `combined_embeddings` (Full Prefix + Suffix).
        # So `prefix_pre_logits` contains the last token embedding too.
        # I should take `prefix_pre_logits[:, :-1]` to match Pi0FAST input?
        # No, `prefix_pre_logits` is the output.
        # Output at pos `t` predicts `t+1`.
        # We want outputs at `tokenized_prompt[:-1]`.
        # These are at indices `[Images_Len : Prefix_Len - 1]`.
        # Length = `Prefix_Len - 1 - Images_Len` = `Prompt_Len - 1`.
        # Matches `targets.shape[1]`.
        
        # We want outputs from inputs corresponding to tokenized_prompt[:-1].
        # These are at the end of the prefix, specifically:
        # Indices: [-(num_targets + 1) : -1]
        # Example: Prompt length 4: [A, B, C, D]
        # Targets: [B, C, D] (length 3)
        # Inputs causing B,C,D: A, B, C
        # Indices of A, B, C relative to end: -4, -3, -2
        # Slice: [-4 : -1]
        
        relevant_pre_logits = prefix_pre_logits[:, -(num_targets + 1) : -1]
        
        action_logits, _ = self.PaliGemma.llm(
            pre_logits=relevant_pre_logits,
        )
        
        logp = jax.nn.log_softmax(action_logits, axis=-1)
        assert observation.token_loss_mask is not None
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        action_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
        action_loss = jnp.mean(action_loss)
        
        # 6. Compute Track Loss (on Suffix part)
        if target_tracks is not None:
            suffix_out = pre_logits[:, prefix_len:] # (B, S, Emb)
            v_t = self.track_out_proj(suffix_out)
            track_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            track_loss = jnp.mean(track_loss)
        else:
            track_loss = jnp.array(0.0, dtype=action_loss.dtype)

        total_loss = self.config.action_loss_weight * action_loss + self.config.track_loss_weight * track_loss
        if return_metrics:
            return total_loss, {
                "loss_total": total_loss,
                "loss_action": action_loss,
                "loss_track": track_loss,
            }
        return total_loss

    def _predict_track_velocity(
        self,
        observation: _model.Observation,
        noisy_tracks: at.Float[at.Array, "b h d"],
        time: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, "b h d"]:
        # Reuse the same backbone + track head pathway as training.
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        batch_size = input_token_embeddings.shape[0]

        track_suffix_tokens = self.embed_track_suffix(None, noisy_tracks, time)
        combined_embeddings = jnp.concatenate([input_token_embeddings, track_suffix_tokens], axis=1)

        prefix_attn_mask = _pi0_fast.make_attn_mask(input_mask, ar_mask)
        action_mask = observation.token_loss_mask
        if action_mask is None:
            prompt_mask = input_mask
        else:
            num_image_tokens = input_mask.shape[1] - action_mask.shape[1]
            if num_image_tokens > 0:
                padding = jnp.zeros((batch_size, num_image_tokens), dtype=bool)
                full_action_mask = jnp.concatenate([padding, action_mask], axis=1)
            else:
                full_action_mask = action_mask
            prompt_mask = jnp.logical_and(input_mask, jnp.logical_not(full_action_mask))

        suffix_len = self.track_horizon
        suffix_to_prefix = einops.repeat(prompt_mask, "b p -> b s p", s=suffix_len)
        suffix_to_suffix = jnp.ones((batch_size, suffix_len, suffix_len), dtype=bool)
        suffix_mask = jnp.concatenate([suffix_to_prefix, suffix_to_suffix], axis=2)
        prefix_to_suffix = jnp.zeros((batch_size, input_mask.shape[1], suffix_len), dtype=bool)
        combined_attn_mask = jnp.concatenate(
            [jnp.concatenate([prefix_attn_mask, prefix_to_suffix], axis=2), suffix_mask], axis=1
        )

        # Consistent position encoding with compute_loss
        prefix_positions = jnp.cumsum(input_mask, axis=1) - 1
        max_prefix_pos = jnp.max(prefix_positions, axis=1, keepdims=True)
        suffix_positions = max_prefix_pos + 1 + jnp.arange(suffix_len)[None, :]
        combined_positions = jnp.concatenate([prefix_positions, suffix_positions], axis=1)

        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=combined_embeddings,
            mask=combined_attn_mask,
            positions=combined_positions,
            return_prelogits=True,
        )
        suffix_out = pre_logits[:, input_token_embeddings.shape[1] :]
        return self.track_out_proj(suffix_out)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> at.Float[at.Array, "b h d"]:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        batch_size = observation.tokenized_prompt.shape[0]
        x_t = jax.random.normal(rng, (batch_size, self.track_horizon, self.track_dim))
        dt = 1.0 / float(num_steps)

        for i in range(int(num_steps)):
            t = 1.0 - i * dt
            t_arr = jnp.full((batch_size,), t, dtype=x_t.dtype)
            v_t = self._predict_track_velocity(observation, x_t, t_arr)
            x_t = x_t - dt * v_t

        return x_t
