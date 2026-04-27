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

from wordle.pattern import PATTERN_SOLVED, PatternMatrix


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

    def update(self, guess: str, pattern: int, pm: PatternMatrix) -> GameState:
        """
        Return the successor state after observing *pattern* for *guess*.

        Filters `candidates` to only those words whose pattern against
        *guess* matches the observed pattern.
        """
        gi = pm._dist._word_to_idx[guess]
        new_candidates = frozenset(
            c for c in self.candidates
            if pm.matrix[gi, c] == pattern
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

    def show(self) -> None:
        """
        Print the game board as coloured Wordle tiles, e.g.:

            C R A N E
            ─────────
            S L A T E   ⬛🟨🟩⬛🟩
            C R A N E   🟩🟩🟩🟩🟩  ✓

        Each letter is rendered on its colour background (green / yellow / grey),
        followed by the emoji row for quick scanning.
        Prints a status line with guess count and remaining candidates.
        """
        from rich.console import Console
        from rich.text import Text

        _BG = {0: "white on grey23", 1: "black on yellow3", 2: "black on green"}
        _EMOJI = {0: "⬛", 1: "🟨", 2: "🟩"}

        console = Console()
        console.print()

        if not self.history:
            console.print("  [dim](no guesses yet)[/dim]")
        else:
            for guess, pattern in self.history:
                digits = []
                p = pattern
                for _ in range(5):
                    digits.append(p % 3)
                    p //= 3
                digits.reverse()

                row = Text("  ")
                for letter, d in zip(guess.upper(), digits):
                    row.append(f" {letter} ", style=_BG[d])
                row.append("  " + "".join(_EMOJI[d] for d in digits))
                console.print(row)

        if self.solved:
            status = f"[green]Solved in {self.guess_count} guess{'es' if self.guess_count != 1 else ''}[/green]"
        elif self.failed:
            status = "[red]Failed — out of guesses[/red]"
        else:
            status = f"Guess {self.guess_count + 1}/{self.max_guesses} — [cyan]{self.remaining}[/cyan] candidates remaining"

        console.print(f"\n  {status}\n")

    def __repr__(self) -> str:
        return (
            f"GameState(guesses={self.guess_count}/{self.max_guesses}, "
            f"remaining={self.remaining}, "
            f"solved={self.solved})"
        )
