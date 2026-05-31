"""
Word vocabulary and probability distribution over answers.

Word lists are sourced from `wordfreq`:
  - `zipf_frequency(word, lang)`: Zipf-scale score (0–7, higher = more common).

Design: one list, every word is both a valid guess and a valid answer.
`WordDistribution` is the single class; `from_wordfreq()` is the main
entry point for loading the vocabulary from the corpus.

Zipf thresholds (measured):
  4.0 →    12 words   (too few)
  3.0 →   223 words
  2.0 → 1 121 words
  1.0 → 3 527 words   ← default (MIN_ZIPF)
"""

from __future__ import annotations

import numpy as np
from wordfreq import get_frequency_dict, zipf_frequency


class WordDistribution:
    """
    A probability distribution over a vocabulary of 5-letter words.

    Primary storage is a dict {word: probability} for O(1) lookups.
    The word list and weight array are derived once at construction for
    fast numpy sampling.  A word-to-index mapping is also maintained so
    that other classes (PatternMatrix) can share it without rebuilding.

    Construction
    ------------
    Pass either a dict of unnormalised weights, or a plain word list:

        WordDistribution({"crane": 2.0, "slate": 1.0})  # custom weights
        WordDistribution(["crane", "slate"])             # uniform
        WordDistribution(["crane", "slate"], zipf=True)  # Zipf-weighted

    Or use the corpus-backed classmethods:

        WordDistribution.from_wordfreq()   # load from wordfreq (weighted)
        WordDistribution.default()         # shorthand for from_wordfreq()
    """

    MIN_ZIPF: float = 1.0

    def __init__(
        self,
        source: dict[str, float] | list[str],
        *,
        zipf: bool = False,
    ):
        """
        Parameters
        ----------
        source : dict[str, float] | list[str]
            Unnormalised weight dict, or a plain word list.
        zipf : bool
            Only used when source is a list.
            False (default) → uniform weights.
            True            → weight ∝ 10^zipf_frequency(word).
        """
        if isinstance(source, dict):
            raw = source
        elif zipf:
            raw = {w: 10 ** zipf_frequency(w, "en") for w in source}
        else:
            raw = {w: 1.0 for w in source}

        if not raw:
            raise ValueError("WordDistribution requires at least one word.")

        total = sum(raw.values())
        # Sort alphabetically so the word order — and therefore the PatternMatrix
        # cache hash — is identical regardless of how the distribution was created
        # (wordfreq frequency order, list insertion order, etc.).
        self._probs: dict[str, float] = {w: raw[w] / total for w in sorted(raw)}
        self._words: list[str] = list(self._probs)
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(self._words)}
        self._weights: np.ndarray = np.array(list(self._probs.values()), dtype=np.float64)

    # ------------------------------------------------------------------
    # Corpus-backed constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_wordfreq(cls, min_zipf: float = MIN_ZIPF, lang: str = 'en') -> WordDistribution:
        """
        Load all 5-letter words with Zipf score >= min_zipf from the wordfreq
        corpus for the given language, weighted by corpus frequency.

        lang: BCP 47 language code supported by wordfreq (e.g. 'en', 'de').

        Note: the word universe is wordfreq's "large" wordlist. Words not
        in that list will not appear even if zipf_frequency() would return a
        qualifying score for them.
        """
        freq_dict = get_frequency_dict(lang, wordlist="large")
        threshold = 10 ** (-7 + min_zipf)
        return cls({
            w: f
            for w, f in freq_dict.items()
            if len(w) == 5 and w.isalpha() and w.islower() and f >= threshold
        })

    @classmethod
    def default(cls, lang: str = 'en') -> WordDistribution:
        """Load the default vocabulary (MIN_ZIPF = 1.0) for the given language."""
        return cls.from_wordfreq(lang=lang)

    # ------------------------------------------------------------------
    # Lookup  (O(1))
    # ------------------------------------------------------------------

    def contains(self, word: str) -> bool:
        """Return True if word is in this distribution."""
        return word in self._probs

    def probability(self, word: str) -> float:
        """Return P(word), or 0.0 if word is not in this distribution."""
        return self._probs.get(word, 0.0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator | int | None = None) -> str:
        """
        Draw a single word according to the distribution.

        Parameters
        ----------
        rng : np.random.Generator | int | None
            Generator, integer seed (reproducible), or None (random).
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(self._words), p=self._weights)
        return self._words[idx]

    # ------------------------------------------------------------------
    # Standard interfaces
    # ------------------------------------------------------------------

    @property
    def words(self) -> list[str]:
        """Ordered list of words in this distribution."""
        return self._words

    def __len__(self) -> int:
        return len(self._probs)

    def __repr__(self) -> str:
        return f"WordDistribution(n={len(self._probs)})"
