"""Text prompt cleaning and embedding (migrated from utils/compute_embedding.py)."""

from __future__ import annotations

import html

import ftfy
import regex as re
import torch


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    return whitespace_clean(basic_clean(text))


def encode_prompt(text_encoder, tokenizer, prompt):
    device = text_encoder.device
    dtype = text_encoder.dtype
    num_videos_per_prompt = 1
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(text_encoder, tokenizer, prompt)
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(text_encoder, tokenizer, prompt)
    return prompt_embeds


def precompute_prompt_embeddings(
    tokenizer,
    text_encoder,
    prompts,
    device,
    dtype,
    batch_size: int = 8,
):
    """Encode unique prompts in batches and keep the cache on CPU."""
    unique_prompts = list(dict.fromkeys(prompts))
    cache = {}
    for start in range(0, len(unique_prompts), batch_size):
        batch = unique_prompts[start : start + batch_size]
        embeddings = (
            compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                batch,
                512,
                device,
                dtype,
                requires_grad=False,
            )
            .detach()
            .to("cpu")
        )
        for prompt, embedding in zip(batch, embeddings):
            cache[prompt] = embedding.contiguous()
    return cache


def gather_prompt_embeddings(cache, prompts, device, dtype):
    """Gather a prompt batch from a CPU embedding cache."""
    return torch.stack([cache[prompt] for prompt in prompts]).to(device=device, dtype=dtype)
