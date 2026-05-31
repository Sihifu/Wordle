"""
Policy interface and policies for Wordle.

A policy is a callable:  (state: GameState, game: WordleGame) -> str

Mathematical utilities (module-level)
--------------------------------------
pattern_marginal   — P(f | g) = Σ_w P(w) · 1{φ(g,w)=f}
entropy            — H(p) = -Σ_k p_k log₂ p_k   (general cost function)
bayesian_update    — P(w | f, g) ∝ P(w) · 1{φ(g,w)=f}

Policies implemented here
--------------------------
RandomPolicy    — picks uniformly at random from remaining candidates.
HumanPolicy     — reads a guess from stdin; renders the board with rich.
EntropyPolicy   — greedy Shannon-entropy maximisation with Bayesian prior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

from .pattern import PatternMatrix
from .game import GameState

if TYPE_CHECKING:
    from .game import WordleGame

console = Console()


# ---------------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------------

def pattern_marginal(
    guess_idx: int,
    candidates: np.ndarray,
    prior: np.ndarray,
    pm: PatternMatrix,
) -> np.ndarray:
    """
    Marginal distribution over feedback patterns for a given guess.

        P(f | g) = Σ_{w ∈ S}  P(w) · 1{φ(g, w) = f}

    With a uniform prior this reduces to counting candidates per pattern.

    Parameters
    ----------
    guess_idx  : row index of the guess in the pattern matrix.
    candidates : shape (n_cands,), indices into the pattern matrix columns.
    prior      : shape (n_cands,), P(w) for each candidate, must sum to 1.
    pm         : PatternMatrix.

    Returns
    -------
    np.ndarray, shape (243,), sums to 1.0.
    """
    patterns = pm.matrix[guess_idx, candidates]          # shape (n_cands,)
    return np.bincount(patterns, weights=prior, minlength=243)


def entropy(distribution: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution (in bits).

        H(p) = -Σ_k  p_k · log₂ p_k

    Parameters
    ----------
    distribution : non-negative array, must sum to 1.

    Returns
    -------
    float — entropy in bits.
    """
    p = distribution[distribution > 0]
    return float(-np.dot(p, np.log2(p)))


def bayesian_update(
    prior: np.ndarray,
    candidates: np.ndarray,
    guess_idx: int,
    observed_pattern: int,
    pm: PatternMatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bayesian posterior update after observing feedback pattern f for guess g.

        P(w | f, g) = P(w) · 1{φ(g,w) = f} / P(f | g)

    Concretely:
      1. Zero out candidates inconsistent with the observed pattern.
      2. Renormalise the remaining probabilities.

    Parameters
    ----------
    prior            : shape (n_cands,), current P(w), sums to 1.
    candidates       : shape (n_cands,), current candidate indices.
    guess_idx        : row index of the guess in the pattern matrix.
    observed_pattern : the pattern integer returned by the game.
    pm               : PatternMatrix.

    Returns
    -------
    (new_candidates, new_prior)
        new_candidates : np.ndarray of surviving candidate indices.
        new_prior      : np.ndarray of normalised posterior probabilities.
    """
    patterns = pm.matrix[guess_idx, candidates]
    mask = patterns == observed_pattern
    new_candidates = candidates[mask]
    new_prior = prior[mask]
    new_prior = new_prior / new_prior.sum()
    return new_candidates, new_prior


# ---------------------------------------------------------------------------
# Policy ABC
# ---------------------------------------------------------------------------

class Policy(ABC):
    """Abstract base class for Wordle policies."""

    @abstractmethod
    def __call__(self, state: GameState, game: WordleGame) -> str:
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
        str — a valid guess word.
        """
        ...


# ---------------------------------------------------------------------------
# Baseline: random policy
# ---------------------------------------------------------------------------

class RandomPolicy(Policy):
    """
    Selects uniformly at random from the remaining candidate answers.

    Simplest non-trivial policy — always guesses a word that could still be
    the answer, but makes no attempt to maximise information.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()

    def __call__(self, state: GameState, game: WordleGame) -> str:
        candidates = list(state.candidates)
        idx = self._rng.choice(len(candidates))
        return game.words[candidates[idx]]

    def __repr__(self) -> str:
        return "RandomPolicy()"


# ---------------------------------------------------------------------------
# Interactive: human policy
# ---------------------------------------------------------------------------

class HumanPolicy(Policy):
    """
    Reads guesses from stdin and renders the board after each turn.

    Validates that the entered word is in the game's word list and
    re-prompts on invalid input.
    """

    def __call__(self, state: GameState, game: WordleGame) -> str:
        state.show()
        while True:
            raw = console.input("  Your guess: ").strip().lower()
            if len(raw) != 5:
                console.print("  [red]Word must be exactly 5 letters.[/red]")
                continue
            if not game.is_valid_word(raw):
                console.print(f"  [red]'{raw}' is not in the word list.[/red]")
                continue
            return raw

    def __repr__(self) -> str:
        return "HumanPolicy()"


# ---------------------------------------------------------------------------
# Entropy policy
# ---------------------------------------------------------------------------

class EntropyPolicy(Policy):
    """
    Greedy Shannon-entropy maximising policy with Bayesian prior.

    At each step:
      1.  Derive the posterior P(w | history) by restricting the initial
          distribution to surviving candidates and renormalising:

              P(w | history) ∝ P₀(w)   for w ∈ candidates

      2.  For every possible guess g, compute the marginal over patterns:

              P(f | g) = Σ_{w ∈ candidates}  P(w | history) · 1{φ(g,w) = f}

      3.  Pick the guess that maximises entropy:

              g* = argmax_g  H( P(· | g) )

      4.  After observing feedback f, the next posterior is obtained via
          `bayesian_update` — zero out inconsistent candidates, renormalise.

    With a uniform initial distribution this is equivalent to the classic
    entropy heuristic that counts candidates per pattern.  With a
    frequency-weighted distribution (e.g. WordDistribution.from_wordfreq)
    common words contribute more to the pattern probabilities, reflecting
    the real-world prior that common words are more likely answers.

    Tie-breaking: among equal-entropy guesses, prefer words that are still
    candidates — they carry the same information but may already be the answer.

    Complexity per step: O(|words| · |candidates|) — fully vectorised.
    """

    def __call__(self, state: GameState, game: WordleGame) -> str:
        candidates = np.array(sorted(state.candidates))
        n = len(candidates)

        if n == 1:
            return game.words[int(candidates[0])]

        # ── posterior over surviving candidates ────────────────────────────
        # P(w | history) ∝ P₀(w) restricted to candidates and renormalised.
        # Falls back to uniform if the distribution assigns zero mass.
        probs = np.array([
            game.pm.distribution.probability(game.words[c]) for c in candidates
        ])
        if probs.sum() == 0:
            probs = np.ones(n)
        prior = probs / probs.sum()

        # ── vectorised P(f | g) for every guess at once ────────────────────
        # sub[i, j] = φ(words[i], candidates[j]),  shape (n_words, n_cands)
        sub = game.pm.matrix[:, candidates].astype(np.int32)
        n_words = sub.shape[0]

        # Offset row i by i*243 so (word, pattern) → unique integer index,
        # then a single weighted bincount gives all marginals simultaneously.
        offsets = np.arange(n_words)[:, np.newaxis] * 243
        flat_idx = (sub + offsets).ravel()
        flat_weights = np.tile(prior, n_words)   # prior repeated per word
        marginals = np.bincount(
            flat_idx, weights=flat_weights, minlength=n_words * 243
        ).reshape(n_words, 243)                  # shape (n_words, 243)

        # ── H(F | g) for each guess ────────────────────────────────────────
        p = marginals
        log_p = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
        entropies = -np.sum(p * log_p, axis=1)   # shape (n_words,)

        # ── tie-break: prefer candidates ───────────────────────────────────
        is_candidate = np.zeros(n_words, dtype=bool)
        is_candidate[candidates] = True
        scores = entropies + is_candidate * 1e-9

        best_gi = int(np.argmax(scores))
        return game.words[best_gi]

    def __repr__(self) -> str:
        return "EntropyPolicy()"
