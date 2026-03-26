"""
Pattern computation for Wordle.

A pattern encodes the coloured feedback for a (guess, answer) pair as a
base-3 integer in [0, 242]:
  - digit value 0  →  grey   (letter absent)
  - digit value 1  →  yellow (letter present, wrong position)
  - digit value 2  →  green  (letter correct)

The most-significant digit corresponds to position 0 (leftmost letter).
PATTERN_SOLVED = 2·81 + 2·27 + 2·9 + 2·3 + 2 = 242.
"""

from __future__ import annotations

import numpy as np
from typing import List

PATTERN_SOLVED = 242  # all-green: every position correct

_SYMBOLS = {0: "⬛", 1: "🟨", 2: "🟩"}


def compute_pattern(guess: str, answer: str) -> int:
    """Return the pattern integer for (guess, answer)."""
    result = [0] * 5
    pool = list(answer)

    # First pass: greens
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2
            pool[i] = None  # consumed

    # Second pass: yellows
    for i in range(5):
        if result[i] == 0 and guess[i] in pool:
            result[i] = 1
            pool[pool.index(guess[i])] = None  # consume one occurrence

    return result[0] * 81 + result[1] * 27 + result[2] * 9 + result[3] * 3 + result[4]


def decode_pattern(pattern: int) -> str:
    """Return an emoji string representation of a pattern integer."""
    digits = []
    p = pattern
    for _ in range(5):
        digits.append(p % 3)
        p //= 3
    return "".join(_SYMBOLS[d] for d in reversed(digits))


def build_pattern_matrix(guesses: List[str], answers: List[str]) -> np.ndarray:
    """
    Precompute all (guess, answer) patterns.

    Returns an array of shape (n_guesses, n_answers) with dtype uint8,
    where entry [i, j] = compute_pattern(guesses[i], answers[j]).
    """
    n_g, n_a = len(guesses), len(answers)
    matrix = np.empty((n_g, n_a), dtype=np.uint8)
    for i, g in enumerate(guesses):
        for j, a in enumerate(answers):
            matrix[i, j] = compute_pattern(g, a)
    return matrix