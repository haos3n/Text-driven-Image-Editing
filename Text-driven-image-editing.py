#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTokenizer

# ----------------------------
# Utility: seed everything
# ----------------------------
def set_seed(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ----------------------------
# Tokenization + alignment utils
# ----------------------------
def tokenize(tokenizer: CLIPTokenizer, prompt: str) -> List[str]:
    """
    Return a list of "basic tokens" (lowercased, stripped) aligned with CLIP tokenization.
    We use tokenizer._tokenizer (BPE) pieces and map back to readable forms heuristically.
    For alignment across prompts, we do a simple whitespace split fallback.
    """
    # Heuristic: prefer whitespace split for human-aligned "tokens" to detect common words
    # Then we will use tokenizer to get token ids for index masking inside attention.
    return [t.strip().lower() for t in prompt.strip().split() if t.strip()]

def common_tokens(src_tokens: List[str], edt_tokens: List[str]) -> Set[str]:
    return set(src_tokens).intersection(set(edt_tokens))

def find_token_positions(tokenizer: CLIPTokenizer, prompt: str, target_word: str) -> List[int]:
    """
    Find positions (in text encoder token indices) where a word appears.
    Note: CLIP BPE may split words; we do a coarse matching on decoded tokens.
    """
    # Tokenize with special tokens
    out = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = out["input_ids"][0].tolist()
    # decode each token id to text piece
    tokens_decoded = [tokenizer.decode([tid]).strip().lower() for tid in input_ids]
    # coarse search: if target_word is substring of decoded piece or exact equal
    # We ignore special tokens "" etc.
    positions = []
    for i, piece in enumerate(tokens_decoded):
        if not piece or piece in ("", "</w>"):
            continue
        if target_word.lower() in piece:
            positions.append(i)
    return positions


def map_shared_token_positions(tokenizer: CLIPTokenizer, src_prompt: str, edt_prompt: str) -> List[Tuple[int, int]]:
    """
    Build a mapping of positions (src_idx, edt_idx) for shared words by greedy left-to-right matching.
    """
    src_words = tokenize(tokenizer, src_prompt)
    edt_words = tokenize(tokenizer, edt_prompt)
    shared = common_tokens(src_words, edt_words)
    if not shared:
        return []

    # Build position lists per word (whitespace tokenization)
    def word_positions(words: List[str]) -> Dict[str, List[int]]:
        pos = {}
        for i, w in enumerate(words):
            pos.setdefault(w, []).append(i)
        return pos

    src_pos_map = word_positions(src_words)
    edt_pos_map = word_positions(edt_words)

    pairs: List[Tuple[int, int]] = []
    for w in shared:
        for i, j in zip(src_pos_map[w], edt_pos_map[w]):
            pairs.append((i, j))
    # Sort by src position
    pairs.sort(key=lambda x: x[0])
    return pairs


# ----------------------------
# Attention Processors
# ----------------------------
class CrossAttentionStore(AttnProcessor):
    """
    A processor that stores cross-attention maps during the forward pass.

    We store attn_probs (softmax(QK^T/sqrt(d))) for cross-attention layers only.
    Key dimension corresponds to text tokens; query corresponds to spatial positions.
    """

    def __init__(self, is_cross_key: str, step_getter: Callable[[], int], store_dict: Dict[str, List[torch.Tensor]]):
        super().__init__()
        self.is_cross_key = is_cross_key  # unique layer key
        self.get_step = step_getter
        self.store = store_dict  # map layer_key -> list of attn tensors per step

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        is_cross = encoder_hidden_states is not None
        bsz, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        if is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # scale
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.baddbmm(
            torch.zeros(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Store cross-attention maps only
        if is_cross:
            step = self.get_step()
            key_name = f"{self.is_cross_key}"  # combine layer identity
            self.store.setdefault(key_name, [])
            # Save a lightweight copy (beware of memory). We detach and maybe cpu() if needed.
            self.store[key_name].append(attn_probs.detach())

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CrossAttentionInjector(AttnProcessor):
    """
    A processor that injects (replaces/overlays/scales) attention probabilities at early steps.

    Modes supported by providing a 'policy' callable:
    - replace: for tokens that changed, use source attn maps (from store) to preserve layout
    - refine: for shared tokens, inject source maps
    - reweight: scale specific token indices by a factor
    """

    def __init__(
        self,
        is_cross_key: str,
        step_getter: Callable[[], int],
        policy: Callable[[torch.Tensor, str, int], torch.Tensor],
    ):
        super().__init__()
        self.is_cross_key = is_cross_key
        self.get_step = step_getter
        self.policy = policy  # fn(attn_probs, layer_key, step) -> attn_probs_modified

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        is_cross = encoder_hidden_states is not None

        query = attn.to_q(hidden_states)
        if is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.baddbmm(
            torch.zeros(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        if is_cross:
            step = self.get_step()
            layer_key = f"{self.is_cross_key}"
            try:
                attn_probs = self.policy(attn_probs, layer_key, step)
            except Exception as e:
                # Robust fallback
                pass

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ----------------------------
# Layer name utilities (identify cross-attn layers)
# ----------------------------
def iter_unet_cross_attn_modules(unet) -> List[Tuple[str, Any]]:
    """
    Iterate all attention modules in UNet and yield (qualified_name, attn_module) for cross-attention places.
    diffusers UNet2DConditionModel has multiple Transformer2DModel blocks with Attention modules.
    """
    # Conservative traversal: find modules that have 'attn1', 'attn2' or '.to_q' etc.
    modules = []
    for name, module in unet.named_modules():
        # We hook at Attention objects that have to_q / to_k / to_v attributes (diffusers Attention)
        has_q = hasattr(module, "to_q")
        has_k = hasattr(module, "to_k")
        has_v = hasattr(module, "to_v")
        has_out = hasattr(module, "to_out")
        if has_q and has_k and has_v and has_out:
            modules.append((name, module))
    return modules


# ----------------------------
# Policies for injection/scaling
# ----------------------------
class ReplaceRefinePolicy:
    """
    A policy used by CrossAttentionInjector for 'replace' and 'refine' modes.

    It uses stored source attention maps for the same layer and same step index (if within inject_steps).
    The actual per-token replacement mask is computed from token alignment:
      - 'replace': inject for tokens that CHANGED (i.e., not common between prompts)
      - 'refine' : inject for tokens that are SHARED between prompts
    """

    def __init__(
        self,
        store: Dict[str, List[torch.Tensor]],
        tokenizer: CLIPTokenizer,
        src_prompt: str,
        edt_prompt: str,
        inject_steps: int,
        mode: str,  # "replace" | "refine"
    ):
        self.store = store
        self.tokenizer = tokenizer
        self.src_prompt = src_prompt
        self.edt_prompt = edt_prompt
        self.inject_steps = max(0, int(inject_steps))
        self.mode = mode

        self.src_words = tokenize(tokenizer, src_prompt)
        self.edt_words = tokenize(tokenizer, edt_prompt)
        self.shared = common_tokens(self.src_words, self.edt_words)
        self.changed: Set[str] = set(self.src_words) ^ set(self.edt_words)  # symmetric diff

        # We build coarse per-token boolean masks (length = text_length) for edited prompt
        # If mode=="replace", select indices for "changed" words in edited prompt
        # If mode=="refine", select indices for "shared" words in edited prompt
        self.edt_token_mask = self._build_token_mask_for_edited()

    def _build_token_mask_for_edited(self) -> torch.Tensor:
        # Build mask of shape [text_length], True for tokens to inject
        out = self.tokenizer(self.edt_prompt, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = out["input_ids"][0]
        pieces = [self.tokenizer.decode([tid]).strip().lower() for tid in input_ids.tolist()]

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, piece in enumerate(pieces):
            if not piece or piece in ("", "</w>"):
                continue
            # coarse piece->word hit
            in_changed = any(w in piece for w in self.changed) if self.changed else False
            in_shared = any(w in piece for w in self.shared) if self.shared else False
            if self.mode == "replace" and in_changed:
                mask[i] = True
            elif self.mode == "refine" and in_shared:
                mask[i] = True
        return mask  # shape [text_len]

    def __call__(self, attn_probs: torch.Tensor, layer_key: str, step: int) -> torch.Tensor:
        """
        attn_probs: [B*heads, Q(spatial), K(text_len)]
        We will replace columns K where edt_token_mask=True with stored source maps (same step & layer).
        """
        if step >= self.inject_steps:
            return attn_probs

        if layer_key not in self.store:
            return attn_probs

        src_list = self.store[layer_key]
        if step >= len(src_list):
            return attn_probs
        src_probs = src_list[step].to(attn_probs.device, dtype=attn_probs.dtype)  # [Bh, Q, K]

        # Align K dimension
        k_len = min(src_probs.shape[-1], attn_probs.shape[-1])
        src_probs = src_probs[..., :k_len]
        edt_probs = attn_probs[..., :k_len]

        # Build mask [K] -> [1,1,K] -> broadcast to [Bh,Q,K]
        mask_1d = self.edt_token_mask[:k_len].to(attn_probs.device)
        if not torch.any(mask_1d):
            return attn_probs
        mask = mask_1d.view(1, 1, -1)

        merged = torch.where(mask, src_probs, edt_probs)
        attn_probs = torch.cat([merged, attn_probs[..., k_len:]], dim=-1)
        return attn_probs


class ReweightPolicy:
    """
    Scale the attention columns corresponding to a target word by a given factor, then renormalize.
    """

    def __init__(self, tokenizer: CLIPTokenizer, edited_prompt: str, scale_token: str, scale_factor: float):
        self.tokenizer = tokenizer
        self.edited_prompt = edited_prompt
        self.scale_token = (scale_token or "").strip().lower()
        self.scale_factor = float(scale_factor) if scale_factor is not None else 1.0

        out = tokenizer(edited_prompt, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=tokenizer.model_max_length)
        self.input_ids = out["input_ids"][0]
        self.token_mask = self._build_token_mask()

    def _build_token_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.input_ids, dtype=torch.bool)
        pieces = [self.tokenizer.decode([tid]).strip().lower() for tid in self.input_ids.tolist()]
        for i, piece in enumerate(pieces):
            if not piece or piece in ("", "</w>"):
                continue
            if self.scale_token and self.scale_token in piece:
                mask[i] = True
        return mask

    def __call__(self, attn_probs: torch.Tensor, layer_key: str, step: int) -> torch.Tensor:
        if self.scale_factor == 1.0:
            return attn_probs
        k_len = attn_probs.shape[-1]
        mask_1d = self.token_mask[:k_len].to(attn_probs.device)
        if not torch.any(mask_1d):
            return attn_probs

        # Scale the selected columns and renormalize along K
        scale = torch.ones(k_len, device=attn_probs.device, dtype=attn_probs.dtype)
        scale = scale * (1.0 + (self.scale_factor - 1.0) * mask_1d.float())
        attn_probs = attn_probs * scale.view(1, 1, -1)
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return attn_probs


# ----------------------------
# Pipeline wrapper
# ----------------------------
class SDPipelines:
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: Optional[str] = None, fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer

    def _set_attn_processors(self, attn_proc_factory: Callable[[str], AttnProcessor]):
        new_procs = {}
        # Walk attention modules to get stable keys
        for name, module in iter_unet_cross_attn_modules(self.pipe.unet):
            # attach processor made by factory with readable key
            new_procs[name] = attn_proc_factory(name)
        self.pipe.unet.set_attn_processor(new_procs)

    def _clear_attn_processors(self):
        self.pipe.unet.set_default_attn_processor()

    def generate_with_store(self, prompt: str, steps: int, guidance: float, seed: Optional[int],
                            height: int, width: int, store_dict: Dict[str, List[torch.Tensor]]):
        """
        One pass that stores cross-attention maps at each step.
        """
        # Time-step counter closure
        step_counter = {"t": 0}
        def get_step():
            return step_counter["t"]

        def make_store_proc(layer_key: str):
            return CrossAttentionStore(
                is_cross_key=layer_key,
                step_getter=get_step,
                store_dict=store_dict
            )

        self._set_attn_processors(make_store_proc)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        # Hook into scheduler progression by overriding callback
        def callback_on_step(step: int, timestep: int, latents: torch.FloatTensor):
            step_counter["t"] = step

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=callback_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        self._clear_attn_processors()
        return image

    def generate_with_injector(self, prompt: str, steps: int, guidance: float, seed: Optional[int],
                               height: int, width: int, policy_maker: Callable[[str], AttnProcessor]):
        """
        One pass that injects/overlays/scales cross-attention maps according to a policy.
        """
        step_counter = {"t": 0}
        def get_step():
            return step_counter["t"]

        def make_inject_proc(layer_key: str):
            return CrossAttentionInjector(
                is_cross_key=layer_key,
                step_getter=get_step,
                policy=policy_maker  # will be called with (attn_probs, layer_key, step)
            )

        # policy_maker signature fix: wrap to satisfy processor call API
        def policy(attn_probs, layer_key, step):
            return policy_maker(attn_probs, layer_key, step)

        self._set_attn_processors(lambda lk: CrossAttentionInjector(lk, get_step, policy))

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        def callback_on_step(step: int, timestep: int, latents: torch.FloatTensor):
            step_counter["t"] = step

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=callback_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        self._clear_attn_processors()
        return image


# ----------------------------
# High-level editing functions
# ----------------------------
def run_replace_or_refine(
    mode: str,
    src_prompt: str,
    edt_prompt: str,
    out_path: str,
    steps: int = 50,
    guidance: float = 7.5,
    inject_steps: int = 25,
    seed: Optional[int] = 123,
    height: int = 512,
    width: int = 512,
):
    assert mode in ("replace", "refine")
    set_seed(seed)
    sd = SDPipelines()

    # 1) Pass: store cross-attention from source prompt
    store: Dict[str, List[torch.Tensor]] = {}
    _ = sd.generate_with_store(
        prompt=src_prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        height=height,
        width=width,
        store_dict=store
    )

    # 2) Pass: inject into edited prompt
    policy_obj = ReplaceRefinePolicy(
        store=store,
        tokenizer=sd.tokenizer,
        src_prompt=src_prompt,
        edt_prompt=edt_prompt,
        inject_steps=inject_steps,
        mode=mode
    )

    def policy(attn_probs, layer_key, step):
        return policy_obj(attn_probs, layer_key, step)

    img = sd.generate_with_injector(
        prompt=edt_prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        height=height,
        width=width,
        policy_maker=policy
    )
    img.save(out_path)
    print(f"[{mode}] saved -> {out_path}")


def run_reweight(
    edt_prompt: str,
    scale_token: str,
    scale_factor: float,
    out_path: str,
    steps: int = 50,
    guidance: float = 7.5,
    seed: Optional[int] = 123,
    height: int = 512,
    width: int = 512,
):
    set_seed(seed)
    sd = SDPipelines()

    policy_obj = ReweightPolicy(
        tokenizer=sd.tokenizer,
        edited_prompt=edt_prompt,
        scale_token=scale_token,
        scale_factor=scale_factor
    )

    def policy(attn_probs, layer_key, step):
        return policy_obj(attn_probs, layer_key, step)

    img = sd.generate_with_injector(
        prompt=edt_prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        height=height,
        width=width,
        policy_maker=policy
    )
    img.save(out_path)
    print(f"[reweight] saved -> {out_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Text-driven Image Editing with Cross-Attention Control (single-file)")
    p.add_argument("--mode", type=str, required=True, choices=["replace", "refine", "reweight"],
                   help="Editing mode")
    p.add_argument("--source-prompt", type=str, default="", help="Source prompt (for replace/refine)")
    p.add_argument("--edited-prompt", type=str, default="", help="Edited prompt (target prompt)")
    p.add_argument("--scale-token", type=str, default="", help="Token to scale in reweight mode")
    p.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor in reweight mode (e.g., 1.8)")
    p.add_argument("--steps", type=int, default=50, help="Num inference steps")
    p.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument("--inject-steps", type=int, default=25, help="Inject cross-attn until step (for replace/refine)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--height", type=int, default=512, help="Image height")
    p.add_argument("--width", type=int, default=512, help="Image width")
    p.add_argument("--out", type=str, required=True, help="Output image path")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("replace", "refine"):
        if not args.source_prompt or not args.edited_prompt:
            raise ValueError("replace/refine mode requires --source-prompt and --edited-prompt")
        run_replace_or_refine(
            mode=args.mode,
            src_prompt=args.source_prompt,
            edt_prompt=args.edited_prompt,
            out_path=args.out,
            steps=args.steps,
            guidance=args.guidance_scale,
            inject_steps=args.inject_steps,
            seed=args.seed,
            height=args.height,
            width=args.width
        )
    elif args.mode == "reweight":
        if not args.edited_prompt or not args.scale_token:
            raise ValueError("reweight mode requires --edited-prompt and --scale-token")
        run_reweight(
            edt_prompt=args.edited_prompt,
            scale_token=args.scale_token,
            scale_factor=args.scale_factor,
            out_path=args.out,
            steps=args.steps,
            guidance=args.guidance_scale,
            seed=args.seed,
            height=args.height,
            width=args.width
        )


if __name__ == "__main__":
    main()
