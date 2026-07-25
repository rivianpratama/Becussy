"""Shared inference helpers, so train.py / eval / chat tokenize identically.

transformers 5.x `apply_chat_template(..., return_tensors="pt")` can return a
BatchEncoding (no `.shape`) rather than a bare tensor depending on version and
call shape. We pin the behavior with return_dict=False and normalize either
form to a 2-D LongTensor of input ids, so downstream `ids.shape[1]` slicing is
always valid (report P0 #4).
"""
from __future__ import annotations

import re

# Qwen3-2507 emits reasoning/tool control tokens (<think>...</think>,
# <tool_call>...) that decode as visible text and aren't caught by
# skip_special_tokens. Strip them so responses read cleanly.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_TAGS = re.compile(r"</?(?:think|tool_call|tool_response)>")


def clean_output(text: str) -> str:
    """Remove leaked reasoning/tool-call control tags from a decoded response."""
    text = _THINK_BLOCK.sub("", text)
    text = _STRAY_TAGS.sub("", text)
    return text.strip()


def encode_messages(tokenizer, messages: list[dict], device: str = "cuda"):
    """Return a [1, seq] LongTensor of input ids for a conversation, with the
    assistant generation prompt appended.

    *messages* is a list of {"role": "user"|"assistant", "content": str}. No
    system message is ever added — that is the training convention, and the
    local server / web app must not diverge from it.
    """
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    # return_dict=False yields a tensor on this stack; guard the BatchEncoding
    # case anyway so a minor version bump can't silently break slicing.
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device)


def encode_chat(tokenizer, user_content: str, device: str = "cuda"):
    """Return a [1, seq] LongTensor of input ids for a single user turn,
    with the assistant generation prompt appended."""
    return encode_messages(
        tokenizer, [{"role": "user", "content": user_content}], device=device
    )
