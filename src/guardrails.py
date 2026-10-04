"""
Shared Guardrails
Multi-Agent Customer Support Intelligence Platform

Reusable safety checks for any agent node that sends customer-controlled
text to an external LLM. Currently used by intake_intent.py; import the
same functions in the Response agent (Phase 5) rather than duplicating
this logic there.

KNOWN LIMITATION: the prompt-injection check below is a pattern-based
pre-filter, not a comprehensive defense. It catches known phrasings
(tested and confirmed against the exact case it was built for — the
LLM itself failed to resist this via system-prompt instructions alone,
which is why this deterministic check exists as a first layer). It will
NOT catch rephrasings, typos, or non-English injection attempts.

Patterns were revised after a false-positive was found: the original
bare "act as" phrase matched innocent urgent language ("act as soon
as possible"), misrouting real customer tickets to intent="other".
Replaced with narrower phrasings ("act as a", "act as an", "act as
if you") that still catch real injection attempts without matching
that case — verified against both the false-positive and the original
injection test cases before merging.
"""
import re

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "disregard previous instructions",
    "disregard your instructions",
    "you are now",
    "system prompt",
    "new instructions",
    "follow these instructions instead",
    "act as a",
    "act as an",
    "act as if you",
    "pretend you are",
]


INJECTION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in INJECTION_PATTERNS) + r")\b",
    re.IGNORECASE,
)


def looks_like_prompt_injection(text):
    """Returns True if text contains a known prompt-injection phrasing.
    Case-insensitive, whole-word match (so 'exact assessment' does not trigger 'act as')."""
    if not text or not isinstance(text, str):
        return False
    return bool(INJECTION_RE.search(text))

SENSITIVE_REQUEST_PATTERNS = [
    "password",
    "otp",
    "one-time password",
    "one time password",
    "card number",
    "cvv",
    "pin number",
    "social security",
    "ssn",
]

REQUEST_VERBS = [
    "send", "share", "provide", "tell", "give", "submit", "confirm",
    "email", "text", "forward", "need", "require", "reply with",
]

SENSITIVE_REQUEST_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in REQUEST_VERBS) + r")\s+"
    r"(?:(?:us|me)\s+)?(?:your|the|a|an)\s+(?:[\w-]+\s+){0,2}?"
    r"(?:" + "|".join(re.escape(p) for p in SENSITIVE_REQUEST_PATTERNS) + r")\b",
    re.IGNORECASE,
)


def asks_for_sensitive_info(text):
    """Returns True if text appears to ask the customer for sensitive
    credentials. Used to catch an LLM-drafted reply that requests something
    it never should, despite being told not to — a deterministic backstop,
    not a substitute for the system prompt instruction."""
    if not text or not isinstance(text, str):
        return False
    return bool(SENSITIVE_REQUEST_RE.search(text))


def exceeds_word_limit(text, max_words=200):
    """Returns True if text is suspiciously long for a support reply —
    catches a malformed or runaway LLM generation (e.g. repeating itself
    or echoing its instructions) that structured-output validation alone
    would not reject, since a schema only checks type, not content length."""
    if not text or not isinstance(text, str):
        return False
    return len(text.split()) > max_words