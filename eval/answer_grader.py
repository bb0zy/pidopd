"""Grade extracted final answers only, never the reasoning transcript."""
from functools import lru_cache
import re

from math_verify import LatexExtractionConfig, parse, verify

MAX_ANSWER_CHARS = 2048


def clean(text):
    text = str(text).strip().rstrip(".。").strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def strip_degree(text):
    return re.sub(r"(?:\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|°)\s*$", "", text).strip()


def components(text):
    """Comma-separated roots are unordered; commas inside fractions stay intact.

    Only explicit braces denote sets; parenthesized tuples retain their semantics.
    Thousands grouping is deliberately not inferred: '1,234' is ambiguous.
    """
    if text.startswith(r"\{") and text.endswith(r"\}"):
        text = text[2:-2].strip()
    stack, parts, start = [], [], 0
    pairs = {"}": "{", ")": "(", "]": "["}
    for i, char in enumerate(text):
        if char in "{([":
            stack.append(char)
        elif char in "})]":
            if not stack or stack.pop() != pairs[char]:
                raise ValueError("Unbalanced answer delimiters")
        elif char in ",，" and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    if stack:
        raise ValueError("Unbalanced answer delimiters")
    parts.append(text[start:].strip())
    if not all(parts) or len(parts) > 16:
        raise ValueError("Empty or too many answer components")
    return parts


@lru_cache(maxsize=4096)
def parse_scalar(text):
    # Wrap the complete isolated answer in one LaTeX region; no plain-text
    # extraction fallback that could accept just a stray number from prose.
    if not text or len(text) > MAX_ANSWER_CHARS or "$" in text:
        raise ValueError("Invalid answer length or embedded math delimiters")
    result = parse("$" + text + "$", extraction_config=[LatexExtractionConfig()],
                   fallback_mode="no_fallback", extraction_mode="first_match",
                   parsing_timeout=5)
    if len(result) != 1 or isinstance(result[0], str):
        raise ValueError("Math parser failed or timed out")
    expression = result[0]
    # Current datasets contain closed numeric expressions, not symbolic unknowns.
    if getattr(expression, "free_symbols", set()):
        raise ValueError("Unresolved symbols or non-mathematical answer text")
    return expression


class AnswerGrader:
    def __init__(self, reference):
        self.reference = str(reference)
        gold = clean(reference)
        self.degree_reference = strip_degree(gold) != gold
        if self.degree_reference:
            gold = strip_degree(gold)
        if len(gold) > MAX_ANSWER_CHARS:
            raise ValueError("Reference answer too long")
        self.gold_parts = components(gold)
        # A bad reference must fail before model initialization, not silently mark
        # every model response wrong. Cached across repetitions of the same answer.
        self.gold = [parse_scalar(part) for part in self.gold_parts]

    def grade(self, prediction):
        if prediction is None:
            return {"correct": False, "grading_method": "missing_answer"}
        try:
            text = clean(prediction)
            if self.degree_reference:
                text = strip_degree(text)
            if len(text) > MAX_ANSWER_CHARS:
                raise ValueError("Answer too long")
            parts = components(text)
            if len(parts) != len(self.gold):
                return {"correct": False, "grading_method": "answer_count_mismatch"}
            if parts == self.gold_parts:
                return {"correct": True, "grading_method": "exact"}
            predicted = [parse_scalar(part) for part in parts]
            # Match all roots one-to-one, not math_verify's any-of list semantics.
            edges = [[bool(verify(gold, pred, strict=True, float_rounding=12,
                                  numeric_precision=30, timeout_seconds=5))
                      for pred in predicted] for gold in self.gold]
            owners = {}

            def match(i, seen):
                for j, equal in enumerate(edges[i]):
                    if equal and j not in seen:
                        seen.add(j)
                        if j not in owners or match(owners[j], seen):
                            owners[j] = i
                            return True
                return False

            correct = all(match(i, set()) for i in range(len(self.gold)))
            return {"correct": correct, "grading_method": "math_verify"}
        except Exception as exc:
            return {"correct": False, "grading_method": "parse_or_verification_error",
                    "grading_error": str(exc)[:300]}
