"""
Policy interface and baseline policies for Wordle.

A policy is a callable:  (state: GameState, game: WordleGame) -> str

Policies implemented here
--------------------------
RandomPolicy    — picks uniformly at random from remaining candidates.
HumanPolicy     — reads a guess from stdin; renders the board with rich.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.text import Text

from .pattern import decode_pattern, PATTERN_SOLVED
from .state import GameState

if TYPE_CHECKING:
    from .game import WordleGame

console = Console()

# Colour map for rich rendering
_RICH_COLOURS = {0: "white on grey23", 1: "black on yellow3", 2: "black on green"}


class Policy(ABC):
    """Abstract base class for Wordle policies."""

    @abstractmethod
    def __call__(self, state: GameState, game: "WordleGame") -> str:
        """
        Choose a guess word given the current information state.

        Parameters
        ----------
        state : GameState
            The agent's current belief state (candidates, history, …).
        game : WordleGame
            The simulator; exposes word lists and validity checks.

        Returns
        -------
        str
            A valid guess word.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each new game (override if stateful)."""
        pass


# ---------------------------------------------------------------------------
# Baseline: random policy
# ---------------------------------------------------------------------------

class RandomPolicy(Policy):
    """
    Selects uniformly at random from the remaining candidate answers.

    This is the simplest non-trivial policy: it always guesses a word that
    could still be the answer, but makes no attempt to maximise information.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()

    def __call__(self, state: GameState, game: "WordleGame") -> str:
        candidates = list(state.candidates)
        idx = self._rng.choice(len(candidates))
        return game.words[candidates[idx]]

    def __repr__(self) -> str:
        return "RandomPolicy()"


# ---------------------------------------------------------------------------
# Interactive: human policy
# ---------------------------------------------------------------------------

def _render_board(state: GameState) -> None:
    """Print the guess history as a coloured Wordle board."""
    console.print()
    for guess, pattern in state.history:
        row = Text()
        digits = []
        p = pattern
        for _ in range(5):
            digits.append(p % 3)
            p //= 3
        digits.reverse()
        for letter, colour_key in zip(guess.upper(), digits):
            row.append(f" {letter} ", style=_RICH_COLOURS[colour_key])
        console.print(row)
    console.print()


class HumanPolicy(Policy):
    """
    Reads guesses from stdin and renders the board after each turn.

    Validates that the entered word is in the game's guess list and
    re-prompts on invalid input.
    """

    def __call__(self, state: GameState, game: "WordleGame") -> str:
        _render_board(state)
        console.print(
            f"  Guess {state.guess_count + 1}/{state.max_guesses}  "
            f"— [cyan]{state.remaining}[/cyan] candidates remaining"
        )
        while True:
            raw = console.input("  Your guess: ").strip().lower()
            if len(raw) != 5:
                console.print("  [red]Word must be exactly 5 letters.[/red]")
                continue
            if not game.is_valid_guess(raw):
                console.print(f"  [red]'{raw}' is not in the word list.[/red]")
                continue
            return raw

    def __repr__(self) -> str:
        return "HumanPolicy()"
