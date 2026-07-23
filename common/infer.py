"""Shared inference helpers, so train.py / eval / chat tokenize identically.

transformers 5.x `apply_chat_template(..., return_tensors="pt")` can return a
BatchEncoding (no `.shape`) rather than a bare tensor depending on version and
call shape. We pin the behavior with return_dict=False and normalize either
form to a 2-D LongTensor of input ids, so downstream `ids.shape[1]` slicing is
always valid (report P0 #4).
"""
from __future__ import annotations


def encode_chat(tokenizer, user_content: str, device: str = "cuda"):
    """Return a [1, seq] LongTensor of input ids for a single user turn,
    with the assistant generation prompt appended."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
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
