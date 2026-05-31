"""
Pattern computation for Wordle.

A pattern encodes the coloured feedback for a (guess, answer) pair as a
base-3 integer in [0, 242]:
  - digit value 0  →  grey   (letter absent)
  - digit value 1  →  yellow (letter present, wrong position)
  - digit value 2  →  green  (letter correct)

The most-significant digit corresponds to position 0 (leftmost letter).
PATTERN_SOLVED = 2·81 + 2·27 + 2·9 + 2·3 + 2 = 242.

Primary API
-----------
compute_pattern(g, a)    — pure function, no state.
decode_pattern(p)        — pure function, returns emoji string.
PatternMatrix(dist)      — precomputed n×n matrix with disk caching.
PatternMatrix.default()  — convenience constructor using wordfreq default.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .words import WordDistribution

PATTERN_SOLVED = 242  # all-green: 2·81 + 2·27 + 2·9 + 2·3 + 2

_SYMBOLS = {0: "⬛", 1: "🟨", 2: "🟩"}
_CACHE_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Pure functions  — no state, safe to import anywhere
# ---------------------------------------------------------------------------

def compute_pattern(guess: str, answer: str) -> int:
    """
    Return the Wordle pattern integer for (guess, answer).

    Two-pass algorithm:
      1. Green pass  — exact position matches; consumed from pool.
      2. Yellow pass — remaining letters present elsewhere; consumed once each.
    """
    result = [0] * 5
    pool = list(answer)

    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2
            pool[i] = None

    for i in range(5):
        if result[i] == 0 and guess[i] in pool:
            result[i] = 1
            pool[pool.index(guess[i])] = None

    return result[0] * 81 + result[1] * 27 + result[2] * 9 + result[3] * 3 + result[4]


def decode_pattern(pattern: int) -> str:
    """Return a 5-emoji string representation of a pattern integer."""
    digits = []
    p = pattern
    for _ in range(5):
        digits.append(p % 3)
        p //= 3
    return "".join(_SYMBOLS[d] for d in reversed(digits))


# ---------------------------------------------------------------------------
# PatternMatrix
# ---------------------------------------------------------------------------

class PatternMatrix:
    """
    Precomputed (n × n) Wordle pattern matrix with transparent disk caching.

    Accepts a WordDistribution (from wordle.words) as its vocabulary source —
    this avoids rebuilding the word-to-index mapping that already lives there.
    Entry [i, j] = compute_pattern(words[i], words[j]).

    Construction
    ------------
    PatternMatrix(dist)          — build/load for the given distribution.
    PatternMatrix.default()      — load wordfreq default vocabulary.

    Caching
    -------
    Saved to `<cache_dir>/patterns_<hash>.npy` on first build; loaded
    from disk on subsequent calls.  The hash encodes the word list, so
    the cache is invalidated automatically when the vocabulary changes.
    """

    def __init__(self, dist: WordDistribution, cache_dir: Path | None = None):
        self._dist = dist
        self._cache_dir: Path = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._matrix: np.ndarray = self._load_or_build()

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, cache_dir: Path | None = None) -> PatternMatrix:
        """Build or load a PatternMatrix for the default wordfreq vocabulary."""
        return cls(WordDistribution.default(), cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The (n × n) pattern matrix, dtype uint8."""
        return self._matrix

    @property
    def words(self) -> list[str]:
        """Ordered vocabulary list."""
        return self._dist.words

    @property
    def distribution(self) -> WordDistribution:
        """The underlying WordDistribution."""
        return self._dist

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, guess: str, answer: str) -> int:
        """Return pattern(guess, answer) in O(1)."""
        idx = self._dist._word_to_idx
        return int(self._matrix[idx[guess], idx[answer]])

    def __contains__(self, word: str) -> bool:
        """Enable `word in pm` membership checks."""
        return word in self._dist._word_to_idx

    def __len__(self) -> int:
        return len(self._dist)

    def __repr__(self) -> str:
        return f"PatternMatrix(n={len(self._dist)}, cache={self._cache_dir})"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _word_list_hash(self) -> str:
        content = "\n".join(self._dist.words).encode()
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self._cache_dir / f"patterns_{self._word_list_hash()}.npy"

    def _load_or_build(self) -> np.ndarray:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path()
        if path.exists():
            return np.load(path)
        matrix = self._build()
        np.save(path, matrix)
        return matrix

    def _build(self) -> np.ndarray:
        words = self._dist.words
        n = len(words)
        matrix = np.empty((n, n), dtype=np.uint8)
        for i, g in enumerate(tqdm(words, desc="Building pattern matrix", unit="word")):
            for j, a in enumerate(words):
                matrix[i, j] = compute_pattern(g, a)
        return matrix


if __name__ == "__main__":
    guess, answer = "slate", "crane"
    p = compute_pattern(guess, answer)
    print(f"compute_pattern('{guess}', '{answer}') = {p}  {decode_pattern(p)}")

    pm = PatternMatrix.default()
    print(pm)
    print(f"get('crane','crane') = {pm.get('crane','crane')}  {decode_pattern(pm.get('crane','crane'))}")
