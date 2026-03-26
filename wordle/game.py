"""
Wordle simulator (environment).

WordleGame holds the precomputed pattern matrix and the word list.
Every word is both a valid guess and a valid answer — there is no
separate guess vocabulary.

The simulator knows the secret word; the agent only sees a GameState.

Typical usage
-------------
    game = WordleGame.build()                       # loads words, builds matrix
    state, target = game.new_game()                 # random answer (uniform)
    state, target = game.new_game(word="crane")     # fixed answer
    state, target = game.new_game(distribution=d)   # sample from distribution d

    while not state.done:
        guess = policy(state, game)
        state, pattern, done = game.step(state, guess, target)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pattern import build_pattern_matrix
from .state import GameState
from .words import WordDistribution, load_words


class WordleGame:
    def __init__(
        self,
        words: list[str],
        pattern_matrix: np.ndarray,
        max_guesses: int = 6,
    ):
        self.words = words                         # single vocabulary: guess = answer
        self.pattern_matrix = pattern_matrix       # shape (n_words, n_words)
        self.max_guesses = max_guesses

        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(words)}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        min_zipf: Optional[float] = None,
        max_guesses: int = 6,
        verbose: bool = True,
    ) -> "WordleGame":
        """
        Build a WordleGame by loading words from wordfreq and
        precomputing the full (n_words × n_words) pattern matrix.

        Parameters
        ----------
        min_zipf : float, optional
            Minimum Zipf frequency threshold.  Defaults to words.MIN_ZIPF.
        """
        from .words import MIN_ZIPF
        threshold = min_zipf if min_zipf is not None else MIN_ZIPF

        if verbose:
            print(f"Loading words from wordfreq (Zipf >= {threshold})…")
        words = load_words(threshold)

        if verbose:
            print(f"  {len(words)} words. Building {len(words)}×{len(words)} pattern matrix…")

        matrix = build_pattern_matrix(words, words)

        if verbose:
            print(f"  Done. Matrix shape: {matrix.shape}")

        return cls(words, matrix, max_guesses)

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def new_game(
        self,
        word: Optional[str] = None,
        distribution: Optional[WordDistribution] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[GameState, str]:
        """
        Start a new game.

        Parameters
        ----------
        word : str, optional
            Use this exact word as the secret answer.
        distribution : WordDistribution, optional
            Sample the secret answer from this distribution.
            Defaults to uniform over all words.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        (initial_state, target_word)
            target_word is the secret — pass it to `step`; keep it from
            the agent/policy.
        """
        if word is not None:
            if word not in self._word_to_idx:
                raise ValueError(f"'{word}' is not in the word list.")
            target = word
        else:
            dist = distribution or WordDistribution.uniform(self.words)
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
    ) -> Tuple[GameState, int, bool]:
        """
        Apply a guess and return the successor state.

        Parameters
        ----------
        state  : current information state (agent's view).
        guess  : word the agent guesses.
        target : secret answer (known only to the simulator).

        Returns
        -------
        (new_state, pattern, done)
            new_state : updated GameState — candidates filtered, history extended.
            pattern   : integer in [0, 242] encoding the colour feedback.
            done      : True if the game has ended (solved or out of guesses).
        """
        if state.done:
            raise ValueError("Game is already over.")
        if guess not in self._word_to_idx:
            raise ValueError(f"'{guess}' is not a valid word.")

        guess_idx = self._word_to_idx[guess]
        target_idx = self._word_to_idx[target]
        pattern = int(self.pattern_matrix[guess_idx, target_idx])

        new_state = state.update(guess, guess_idx, pattern, self.pattern_matrix)
        return new_state, pattern, new_state.done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_valid_word(self, word: str) -> bool:
        return word in self._word_to_idx

    def __repr__(self) -> str:
        return (
            f"WordleGame(words={len(self.words)}, "
            f"max_guesses={self.max_guesses})"
        )