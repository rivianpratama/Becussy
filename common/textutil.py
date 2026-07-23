"""Shared text utilities for the topic-engagement heuristic.

Used by dataset/scripts/validate.py (training-data gate) and eval/metrics.py
(checkpoint metric) so the two pipelines measure "engaged with the question"
identically.
"""
from __future__ import annotations

import re

STOPWORDS = set(
    """a an the is are was were be been being do does did to of in on for with as at by
    from that this it its and or but if then so not no how what why who when where which
    can could should would will shall may might must you your i my we our me us they them
    he she his her about into over under out up down there here please write give tell
    me explain describe your best way make good many much very really just like get one
    two three want need know think see say says said also more most some any all
    apa yang bagaimana cara untuk dengan dari dan atau tapi jika saya kamu anda itu ini
    di ke pada adalah bisa harus mau tolong""".split()
)


def content_words(text: str) -> set[str]:
    """Lowercased content words (len > 2, not stopwords). Handles EN and ID."""
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) > 2 and w not in STOPWORDS
    }
