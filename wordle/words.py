"""
Word lists and probability distributions over answers.

Word lists are sourced from `wordfreq`, which provides:
  - A large English lexicon with corpus-based frequency estimates.
  - `zipf_frequency(word, lang)`: Zipf-scale frequency (0–7, higher = more common).

Design: one list, no distinction between answers and guesses
------------------------------------------------------------
Every word in the list is both a valid guess *and* a valid answer.
`load_words()` is the single entry point; `MIN_ZIPF` controls vocabulary size.
Zipf 4.0 ≈ top ~3 000 most common English words, which gives a clean,
recognisable set comparable to the original Wordle answer list.
"""

from __future__ import annotations

import numpy as np
from wordfreq import get_frequency_dict, zipf_frequency
from typing import List, Optional

# Zipf 4.0 ≈ top ~3 000 most common 5-letter English words.
# Raise to shrink the vocabulary; lower to expand it.
MIN_ZIPF: float = 4.0


def load_words(min_zipf: float = MIN_ZIPF) -> List[str]:
    """
    Return sorted 5-letter lowercase English words with Zipf score >= min_zipf.

    Every word in this list is both a valid guess and a valid answer.
    """
    freq_dict = get_frequency_dict("en", wordlist="best")
    return sorted(
        w for w, f in freq_dict.items()
        if len(w) == 5 and w.isalpha() and w.islower() and f >= 10 ** (-7 + min_zipf)
    )


class WordDistribution:
    """
    A probability distribution over a word list.

    Supports uniform sampling and frequency-weighted sampling
    (words proportional to their corpus frequency).
    """

    def __init__(self, words: List[str], weights: Optional[np.ndarray] = None):
        self.words = words
        if weights is None:
            weights = np.ones(len(words), dtype=np.float64)
        self.weights = weights / weights.sum()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def uniform(cls, words: List[str]) -> "WordDistribution":
        """Uniform distribution: every word equally likely."""
        return cls(words)

    @classmethod
    def from_zipf(cls, words: List[str]) -> "WordDistribution":
        """
        Frequency-weighted distribution using Zipf scores from wordfreq.

        Weight of word w  ∝  10^zipf_frequency(w, 'en').
        This gives more common words a higher probability of being the answer,
        mirroring how real Wordle selects its daily words.
        """
        weights = np.array(
            [10 ** zipf_frequency(w, "en") for w in words], dtype=np.float64
        )
        return cls(words, weights)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: Optional[np.random.Generator] = None) -> str:
        """Draw a single word according to the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(self.words), p=self.weights)
        return self.words[idx]

    def probability(self, word: str) -> float:
        """Return the probability assigned to a given word."""
        try:
            idx = self.words.index(word)
            return float(self.weights[idx])
        except ValueError:
            return 0.0

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return f"WordDistribution(n={len(self.words)})"