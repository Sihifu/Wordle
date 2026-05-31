"""
Wordle game state and simulator.

GameState is the agent's belief state: all information accumulated from
observations so far. It is deliberately immutable — every update returns a
fresh GameState — so states can be safely shared, hashed, and used as dict
keys (e.g. for memoisation).

WordleGame owns a PatternMatrix and uses it as the single source of truth
for the vocabulary, word indices, and pattern lookups. Every word is both a
valid guess and a valid answer.

Typical usage
-------------
    game = WordleGame.build()                       # loads words, builds matrix
    state, target = game.new_game()                 # random answer (uniform)
    state, target = game.new_game(word="crane")     # fixed answer

    while not state.done:
        guess = policy(state, game)
        state, pattern, done = game.step(state, guess, target)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .words import WordDistribution
from .pattern import PatternMatrix, PATTERN_SOLVED


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameState:
    candidates: frozenset   # frozenset[int] — remaining answer indices
    history: tuple          # tuple of (guess: str, pattern: int)
    max_guesses: int = 6

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
        return len(self.candidates)

    def update(self, guess: str, pattern: int, pm: PatternMatrix) -> GameState:
        """Return the successor state after observing pattern for guess."""
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

    def last_guess(self) -> str | None:
        return self.history[-1][0] if self.history else None

    def last_pattern(self) -> int | None:
        return self.history[-1][1] if self.history else None

    def show(self) -> None:
        """Print the game board as coloured Wordle tiles."""
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


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class WordleGame:
    def __init__(self, pm: PatternMatrix, max_guesses: int = 6):
        self.pm = pm
        self.words = pm.words
        self.max_guesses = max_guesses
        self._word_to_idx = pm._dist._word_to_idx

    @classmethod
    def build(
        cls,
        min_zipf: float | None = None,
        lang: str = 'en',
        max_guesses: int = 6,
    ) -> WordleGame:
        """Build a WordleGame from the wordfreq corpus."""
        dist = WordDistribution.from_wordfreq(
            min_zipf if min_zipf is not None else WordDistribution.MIN_ZIPF,
            lang=lang,
        )
        return cls(PatternMatrix(dist), max_guesses)

    def new_game(
        self,
        word: str | None = None,
        distribution: WordDistribution | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[GameState, str]:
        """
        Start a new game.

        Parameters
        ----------
        word         : use this exact word as the secret answer.
        distribution : sample the secret from this distribution (default: uniform).
        rng          : random generator for reproducibility.

        Returns (initial_state, target_word).  Keep target_word from the agent.
        """
        if word is not None:
            if word not in self._word_to_idx:
                raise ValueError(f"'{word}' is not in the word list.")
            target = word
        else:
            dist = distribution or WordDistribution(self.words)
            target = dist.sample(rng)

        initial_state = GameState(
            candidates=frozenset(range(len(self.words))),
            history=(),
            max_guesses=self.max_guesses,
        )
        return initial_state, target

    def step(
        self,
        state: GameState,
        guess: str,
        target: str,
    ) -> tuple[GameState, int, bool]:
        """
        Apply a guess and return (new_state, pattern, done).

        pattern : integer in [0, 242] encoding the colour feedback.
        done    : True if the game has ended (solved or out of guesses).
        """
        if state.done:
            raise ValueError("Game is already over.")
        if guess not in self._word_to_idx:
            raise ValueError(f"'{guess}' is not a valid word.")

        pattern = self.pm.get(guess, target)
        new_state = state.update(guess, pattern, self.pm)
        return new_state, pattern, new_state.done

    def is_valid_word(self, word: str) -> bool:
        return word in self._word_to_idx

    def __repr__(self) -> str:
        return f"WordleGame(words={len(self.words)}, max_guesses={self.max_guesses})"
