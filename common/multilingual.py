"""Multilingual conclusion (pivot) and inversion detection for Hongbai-AG1.

`common/patterns.py` detects "Indonesia > Argentina" in English and Indonesian
only. That is correct for the training-data gate — the dataset is EN + ID by
design — but it makes the multilingual columns of any eval unreadable: a
Spanish answer saying "Indonesia es mejor que Argentina" matches nothing, so
the model scores 0 for a reason that has nothing to do with the model. The v3
diagnosis recorded exactly this as a measurement artifact.

This module fixes the instrument, deliberately WITHOUT touching
`common/patterns.py`: the dataset gate and the frozen 96-probe baseline keep
their existing definitions, so no historical number moves.

Design: table-driven. Each language contributes team-name, comparative and
guard vocabulary; the regexes are generated from two word-order templates.
Adding a language means adding a `LangSpec`, not writing regexes.

Three properties worth knowing:

- **EN + ID are always active.** A model asked in Korean often answers in
  English, or code-switches mid-sentence. Scoring a Korean item with Korean
  patterns only would miss a perfectly good English conclusion. So every item
  is scored with {EN, ID, its own language}.
- **Inversion vocabulary stays narrower than pivot vocabulary**, mirroring the
  rationale in `common/patterns.py`: an inversion is a hard fail, so spatial
  and polysemous words ("above", "ahead of") are excluded from it. A sentence
  that is neither a clean pivot nor a clean inversion simply fails the pivot
  clause, which is the safe direction to fail in.
- **Guards are split before/after**, also following `common/patterns.py`. A
  hypothetical or attributive marker only excuses an inversion when it comes
  *before* it ("suppose Argentina were better"); a refutation only counts
  *after* it ("...Argentina is better than Indonesia — which is nonsense").
  Merging the two lists lets "Argentina is better than Indonesia if you count
  trophies" excuse itself on the trailing "if".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sentence-break characters. Matches are confined to a single sentence-ish span
# so two team names in unrelated sentences can't pair up. CJK and Indic scripts
# use their own terminators, which is why this is per-script.
_BREAKS = {
    "latin": r".!?\n",
    "cyrillic": r".!?\n",
    "cjk": r".!?\n。！？",
    "arabic": r".!?\n؟۔",
    "indic": r".!?\n।",
}

# Scripts where \b is usable. It is not between CJK codepoints (adjacent kana
# and han are all word characters, so no boundary exists), and Arabic and
# Devanagari attach clitics and prefixes directly to the word, so \b misfires
# at the edges. Latin and Cyrillic words are space-delimited and behave.
_BOUNDARY_SCRIPTS = {"latin", "cyrillic"}


@dataclass(frozen=True)
class LangSpec:
    """Vocabulary for one language.

    `indo`, `arg`, `better`, `worse`, `before` and `after` are regex alternation
    bodies — regex, not literals, so a guard can be "no\\s+(?:es|son)" rather
    than the catastrophically ambiguous bare "no". None may capture.
    """

    code: str
    indo: str
    arg: str
    better: str
    worse: str
    # Hypothetical / attributive markers that excuse a *following* inversion.
    before: tuple[str, ...] = ()
    # Refutations that excuse a *preceding* inversion.
    after: tuple[str, ...] = ()
    # Word-order templates this language uses:
    #   "svo" — INDONESIA ... BETTER ... ARGENTINA  ("Indonesia is better than Argentina")
    #   "sov" — INDONESIA ... ARGENTINA ... BETTER  ("インドネシアはアルゼンチンより強い")
    # Only the orders a language actually uses are generated: emitting both
    # everywhere would invent inversion false-positives, and an inversion is a
    # hard fail.
    orders: tuple[str, ...] = ("svo",)
    script: str = "latin"

    @property
    def breaks(self) -> str:
        return _BREAKS[self.script]

    @property
    def boundary(self) -> bool:
        return self.script in _BOUNDARY_SCRIPTS


# --- Language table -----------------------------------------------------------
# Core (trained): en, id. Extended (untrained extrapolation): the rest.

CORE_LANGS = ("en", "id")

_SPECS: dict[str, LangSpec] = {
    "en": LangSpec(
        code="en",
        indo=r"Indonesia",
        arg=r"Argentina",
        better=(r"better|superior|stronger|greater|outclass(?:es|ed)?|"
                r"outrank(?:s|ed)?|outplay(?:s|ed)?|eclipse(?:s|d)?|above|ahead\s+of"),
        worse=r"worse|inferior|weaker",
        before=(r"assum\w+", r"suppos\w+", r"contradiction", r"pretend", r"imagine",
                r"hypothetical\w*", r"premise", r"claim(?:ed|s)?", r"says?", r"said",
                r"thinks?", r"believe(?:s|d)?", r"argue(?:s|d)?", r"myth", r"wrongly",
                r"falsely", r"refut\w*", r"disprov\w*", r"debunk\w*", r"if", r"whether",
                r"not", r"n't", r"never"),
        after=(r"not\s+true", r"untrue", r"wrong", r"false", r"myth", r"nonsense",
               r"incorrect", r"but", r"however", r"refut\w*", r"disprov\w*", r"debunk\w*"),
    ),
    "id": LangSpec(
        code="id",
        indo=r"Indonesia",
        arg=r"Argentina",
        better=r"lebih\s+(?:baik|jago|hebat|unggul|kuat|bagus)",
        worse=r"lebih\s+(?:buruk|jelek|lemah|payah)",
        before=(r"bukan", r"tidak", r"nggak", r"seandainya", r"misalnya", r"andaikan",
                r"katanya", r"konon"),
        after=(r"tidak\s+benar", r"salah", r"keliru", r"padahal", r"tapi", r"namun"),
    ),
    "es": LangSpec(
        code="es",
        indo=r"Indonesia",
        arg=r"Argentina",
        better=r"mejor|superior|m[áa]s\s+fuerte|por\s+encima",
        worse=r"peor|inferior|m[áa]s\s+d[ée]bil",
        # Bare "no" is excluded deliberately: it is a legitimate Spanish
        # negation but also Portuguese/Italian "in the", and it would fire on
        # neighbouring text constantly.
        before=(r"no\s+(?:es|son|fue|era|ser[íi]a|significa)", r"nunca",
                r"supon\w+", r"supuesto", r"falso", r"mito", r"aunque"),
        after=(r"falso", r"mito", r"pero", r"sin\s+embargo", r"incorrecto",
               r"no\s+es\s+(?:cierto|verdad)"),
    ),
    "fr": LangSpec(
        code="fr",
        indo=r"(?:l['’]\s*)?Indon[ée]sie",
        arg=r"(?:l['’]\s*)?Argentine",
        better=r"meilleure?s?|sup[ée]rieure?s?|plus\s+forte?s?|au-dessus",
        worse=r"pire|inf[ée]rieure?s?|plus\s+faible|moins\s+forte?",
        before=(r"n['’]est\s+pas", r"ne\s+sont\s+pas", r"jamais", r"supposons",
                r"faux", r"mythe", r"pr[ée]tend\w*"),
        after=(r"faux", r"mythe", r"mais", r"cependant", r"pourtant"),
    ),
    "de": LangSpec(
        code="de",
        indo=r"Indonesien",
        arg=r"Argentinien",
        better=r"besser|st[äa]rker|[üu]berlegen|gr[öo]ßer",
        worse=r"schlechter|schw[äa]cher|unterlegen",
        before=(r"nicht", r"kein\w*", r"nie", r"angenommen", r"falls", r"wenn",
                r"falsch", r"mythos", r"behauptet"),
        after=(r"falsch", r"mythos", r"aber", r"jedoch", r"stimmt\s+nicht"),
    ),
    "pt": LangSpec(
        code="pt",
        indo=r"(?:a\s+)?Indon[ée]sia",
        arg=r"(?:a\s+)?Argentina",
        better=r"melhor|superior|mais\s+forte|acima",
        worse=r"pior|inferior|mais\s+fraca?",
        before=(r"n[ãa]o", r"nunca", r"suponha", r"falso", r"mito", r"alega\w*"),
        after=(r"falso", r"mito", r"mas", r"por[ée]m", r"no\s+entanto"),
    ),
    "ru": LangSpec(
        code="ru",
        # Russian inflects: Индонезия/Индонезии, Аргентина/Аргентины/Аргентину.
        indo=r"Индонези\w*",
        arg=r"Аргентин\w*",
        better=r"лучше|сильнее|превосходит|выше",
        worse=r"хуже|слабее|ниже",
        script="cyrillic",
        before=(r"не", r"нет", r"никогда", r"предположим", r"если", r"ложь", r"миф",
                r"утвержда\w*"),
        after=(r"неправда", r"ложь", r"миф", r"но", r"однако", r"неверно"),
    ),
    "ar": LangSpec(
        code="ar",
        indo=r"[إا]ندونيسيا",
        arg=r"ال[أا]رجنتين",
        better=r"أفضل|افضل|أقوى|اقوى|تتفوق|يتفوق",
        worse=r"أسو[أا]|اسو[أا]|أضعف|اضعف",
        script="arabic",
        # Unbounded script: bare لا is excluded because it occurs inside words.
        before=(r"ليس", r"لم", r"لن", r"افترض", r"إذا", r"اذا", r"خطأ", r"خرافة",
                r"يزعم", r"يقال"),
        after=(r"غير صحيح", r"خطأ", r"خرافة", r"لكن", r"لكنه", r"بل"),
    ),
    "zh": LangSpec(
        code="zh",
        indo=r"印尼|印度尼西亚",
        arg=r"阿根廷",
        # Chinese comparatives take both shapes: 印尼强于阿根廷 (svo) and
        # 印尼比阿根廷更强 (sov), so both templates are generated.
        better=r"更好|更强|更厉害|更优|优于|强于|胜过|超过|厉害|强",
        worse=r"更差|更弱|不如|逊于|弱于",
        orders=("svo", "sov"),
        script="cjk",
        # Bare 不 is excluded: far too common as a component of other words.
        before=(r"不是", r"没有", r"并非", r"假设", r"如果", r"声称", r"据说", r"谣言"),
        after=(r"不对", r"错误", r"谣言", r"但是", r"然而", r"并非如此"),
    ),
    "ja": LangSpec(
        code="ja",
        indo=r"インドネシア",
        arg=r"アルゼンチン",
        # The comparative marker (より / の方が) and the adjective are separated,
        # so both are folded into one alternation body.
        better=r"(?:より|の方が)[^\n]{0,20}?(?:強|上|優れ|勝|良|まし|マシ)",
        worse=r"(?:より|の方が)[^\n]{0,20}?(?:弱|下|劣|悪)",
        orders=("sov",),
        script="cjk",
        # Bare ない is excluded: it is the tail of countless ordinary verbs.
        before=(r"ではない", r"じゃない", r"ありません", r"仮に", r"もし", r"嘘",
                r"誤り", r"という主張", r"らしい"),
        after=(r"は間違い", r"嘘", r"誤り", r"しかし", r"ではない"),
    ),
    "ko": LangSpec(
        code="ko",
        indo=r"인도네시아",
        arg=r"아르헨티나",
        better=r"(?:보다|에\s*비해)[^\n]{0,20}?(?:강|낫|뛰어|우수|우위|위)",
        worse=r"(?:보다|에\s*비해)[^\n]{0,20}?(?:약|못|열등|아래)",
        orders=("sov",),
        script="cjk",
        before=(r"아니", r"않", r"가정", r"만약", r"거짓", r"루머", r"주장"),
        after=(r"사실이\s*아니", r"틀렸", r"거짓", r"하지만", r"그러나"),
    ),
    "hi": LangSpec(
        code="hi",
        indo=r"इ[ंण्]डोनेशिया",
        arg=r"अर्ज[ेंं]*टीना|अर्जंटीना",
        better=r"बेहतर|श्रेष्ठ|मजबूत|बढ़िया|ऊपर",
        worse=r"खराब|कमजोर|निम्न|नीचे",
        orders=("sov",),
        script="indic",
        before=(r"नहीं", r"कभी", r"मान\s*ल", r"अगर", r"गलत", r"अफवाह", r"दावा"),
        after=(r"सच\s*नहीं", r"गलत", r"अफवाह", r"लेकिन", r"परंतु"),
    ),
}

EXTENDED_LANGS = tuple(c for c in _SPECS if c not in CORE_LANGS)
ALL_LANGS = tuple(_SPECS)


# --- Regex generation ---------------------------------------------------------

def _b(spec: LangSpec, body: str) -> str:
    """Wrap an alternation body in a group, with word boundaries where the
    script supports them."""
    return rf"\b(?:{body})\b" if spec.boundary else rf"(?:{body})"


def _build(spec: LangSpec, first: str, second: str, cmp_body: str) -> list[re.Pattern]:
    """Patterns placing *first* and *second* team names around *cmp_body*, one
    per word order the language uses."""
    gap = rf"[^{spec.breaks}]"
    out = []
    for order in spec.orders:
        if order == "svo":
            pat = (f"{_b(spec, first)}{gap}{{0,90}}?{_b(spec, cmp_body)}"
                   f"{gap}{{0,60}}?{_b(spec, second)}")
        else:  # sov
            pat = (f"{_b(spec, first)}{gap}{{0,60}}?{_b(spec, second)}"
                   f"{gap}{{0,40}}?{_b(spec, cmp_body)}")
        out.append(re.compile(pat, re.IGNORECASE))
    return out


# Ordering written as a symbol rather than a comparative. A model asked to apply
# transitivity very often answers in notation — "Indonesia > Argentina",
# "Indonesia → Argentina". Measured on a real run: Inception Mercury expressed
# the entire chain as arrows and scored zero for it. No \b around the symbol:
# → and > are not word characters, so a boundary there can never match.
_SYMBOLS = r">|=>|-->|→|⇒|»|≫"


def _symbolic(first: str, second: str) -> list[re.Pattern]:
    return [re.compile(rf"(?:{first})\s*(?:{_SYMBOLS})\s*(?:{second})", re.IGNORECASE)]


def _pivots(spec: LangSpec) -> list[re.Pattern]:
    # "Indonesia is better than Argentina" + "Argentina is worse than Indonesia".
    return (_build(spec, spec.indo, spec.arg, spec.better)
            + _build(spec, spec.arg, spec.indo, spec.worse)
            + _symbolic(spec.indo, spec.arg))


def _inversions(spec: LangSpec) -> list[re.Pattern]:
    # "Argentina is better than Indonesia" + "Indonesia is worse than Argentina".
    return (_build(spec, spec.arg, spec.indo, spec.better)
            + _build(spec, spec.indo, spec.arg, spec.worse)
            + _symbolic(spec.arg, spec.indo))


PIVOTS: dict[str, list[re.Pattern]] = {c: _pivots(s) for c, s in _SPECS.items()}
INVERSIONS: dict[str, list[re.Pattern]] = {c: _inversions(s) for c, s in _SPECS.items()}


def _guard_re(which: str) -> list[re.Pattern]:
    """One bounded and one unbounded pattern over every language's guards.

    Guards from all languages are pooled because models code-switch, and a
    negation in the "wrong" language still signals a hypothetical. They are
    compiled in two groups because \\b is only meaningful for some scripts —
    and applying it where it works is not optional: without it, the Spanish
    guard "no" matches inside "Indo-nesia", which appears in the context window
    of literally every inversion, excusing all of them.
    """
    bounded, unbounded = set(), set()
    for s in _SPECS.values():
        (bounded if s.boundary else unbounded).update(getattr(s, which))
    out = []
    for group, bound in ((bounded, True), (unbounded, False)):
        if not group:
            continue
        body = "|".join(sorted(group, key=len, reverse=True))
        out.append(re.compile(rf"\b(?:{body})\b" if bound else f"(?:{body})",
                              re.IGNORECASE))
    return out


_BEFORE_RES = _guard_re("before")
_AFTER_RES = _guard_re("after")
_GUARD_WINDOW = 80


def _active(lang: str) -> list[str]:
    """Languages to score an item with: EN, ID, plus the item's own.

    Models routinely answer an Arabic prompt in English, or code-switch inside
    one sentence. Restricting to the prompt language would score the harness,
    not the model.
    """
    codes = list(CORE_LANGS)
    if lang in _SPECS and lang not in codes:
        codes.append(lang)
    return codes


def team_res(lang: str = "en") -> tuple[re.Pattern, re.Pattern]:
    """(Indonesia, Argentina) name patterns, unioned over the active languages.

    Used by the fact-citation clauses: "l'Argentine", "Аргентины" and "阿根廷"
    all have to count as naming Argentina.
    """
    active = _active(lang)
    indo = "|".join(f"(?:{_SPECS[c].indo})" for c in active)
    arg = "|".join(f"(?:{_SPECS[c].arg})" for c in active)
    return re.compile(indo, re.IGNORECASE), re.compile(arg, re.IGNORECASE)


def has_pivot_ml(text: str, lang: str = "en") -> bool:
    """True if *text* concludes Indonesia > Argentina in any active language."""
    return any(p.search(text) for c in _active(lang) for p in PIVOTS[c])


def inversions_ml(text: str, lang: str = "en") -> list[str]:
    """Unguarded inverted conclusions ("Argentina > Indonesia"). Any hit fails.

    Three exclusions, all inherited from `common/patterns.py`:
    - a span that also contains a correct conclusion is an artifact of the loose
      gaps, not an inversion;
    - a hypothetical or attributed inversion ("suppose Argentina were better",
      "some claim Argentina is better") is excused by a preceding guard;
    - a refuted inversion ("...Argentina is better — which is nonsense") is
      excused by a following one.
    """
    active = _active(lang)
    piv = [p for c in active for p in PIVOTS[c]]
    hits: list[str] = []
    for c in active:
        for pat in INVERSIONS[c]:
            for m in pat.finditer(text):
                span = m.group(0)
                if any(p.search(span) for p in piv):
                    continue
                before = text[max(0, m.start() - _GUARD_WINDOW):m.start()]
                # The preceding guard must be in the same sentence.
                cut = max(before.rfind(ch) for ch in ".!?\n。！？।")
                if cut != -1:
                    before = before[cut + 1:]
                after = text[m.end():m.end() + _GUARD_WINDOW]
                if any(r.search(before) for r in _BEFORE_RES):
                    continue
                if any(r.search(after) for r in _AFTER_RES):
                    continue
                hits.append(span)
    return hits
