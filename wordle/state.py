"""
Agent information state for Wordle.

GameState is the agent's *belief state*: all information it has accumulated
from observations so far.  It is deliberately immutable — every update
returns a fresh GameState — so states can be safely shared, hashed, and
used as dict keys (e.g. for solver memoisation).

Key fields
----------
candidates : frozenset[int]
    Indices (into the answers list) of words still consistent with every
    observation received so far.  At game start this is all indices.
history : tuple[tuple[str, int], ...]
    Ordered sequence of (guess_word, pattern) pairs seen so far.
max_guesses : int
    Maximum allowed guesses (6 in standard Wordle).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from .pattern import PATTERN_SOLVED


@dataclass(frozen=True)
class GameState:
    candidates: frozenset          # frozenset[int] — remaining answer indices
    history: Tuple                  # tuple of (guess: str, pattern: int)
    max_guesses: int = 6

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def guess_count(self) -> int:
        return len(self.history)

    @property
    def solved(self) -> bool:
        """True if the last guess produced the all-green pattern."""
        return bool(self.history) and self.history[-1][1] == PATTERN_SOLVED

    @property
    def failed(self) -> bool:
        """True if guesses exhausted without solving."""
        return not self.solved and self.guess_count >= self.max_guesses

    @property
    def done(self) -> bool:
        return self.solved or self.failed

    @property
    def remaining(self) -> int:
        """Number of candidate answers still consistent with observations."""
        return len(self.candidates)

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def update(
        self,
        guess: str,
        guess_idx: int,
        pattern: int,
        pattern_matrix: np.ndarray,
    ) -> "GameState":
        """
        Return the successor state after observing *pattern* for *guess*.

        Filters `candidates` to only those words whose pattern against
        *guess* matches the observed pattern.
        """
        new_candidates = frozenset(
            c for c in self.candidates
            if pattern_matrix[guess_idx, c] == pattern
        )
        return replace(
            self,
            candidates=new_candidates,
            history=self.history + ((guess, pattern),),
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def last_guess(self) -> str | None:
        return self.history[-1][0] if self.history else None

    def last_pattern(self) -> int | None:
        return self.history[-1][1] if self.history else None

    def __repr__(self) -> str:
        return (
            f"GameState(guesses={self.guess_count}/{self.max_guesses}, "
            f"remaining={self.remaining}, "
            f"solved={self.solved})"
        )
