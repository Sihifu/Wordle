"""End-to-end tests for WordleGame simulator."""

import pytest
import numpy as np
from wordle.game import WordleGame
from wordle.pattern import build_pattern_matrix, PATTERN_SOLVED
from wordle.words import WordDistribution
from wordle.policy import RandomPolicy


# ---------------------------------------------------------------------------
# Lightweight fixture — bypasses wordfreq for speed
# ---------------------------------------------------------------------------

WORDS = ["crane", "slate", "audio", "stale", "groan"]

@pytest.fixture(scope="module")
def tiny_game():
    """A 5-word game (every word is both a valid guess and a valid answer)."""
    matrix = build_pattern_matrix(WORDS, WORDS)
    return WordleGame(WORDS, matrix, max_guesses=6)


# ---------------------------------------------------------------------------
# new_game
# ---------------------------------------------------------------------------

class TestNewGame:
    def test_fixed_word(self, tiny_game):
        state, target = tiny_game.new_game(word="crane")
        assert target == "crane"
        assert state.remaining == len(tiny_game.words)
        assert state.guess_count == 0

    def test_unknown_word_raises(self, tiny_game):
        with pytest.raises(ValueError):
            tiny_game.new_game(word="zzzzz")

    def test_uniform_distribution(self, tiny_game):
        rng = np.random.default_rng(42)
        _, target = tiny_game.new_game(rng=rng)
        assert target in tiny_game.words

    def test_custom_distribution(self, tiny_game):
        dist = WordDistribution(["crane", "slate"], np.array([1.0, 0.0]))
        _, target = tiny_game.new_game(distribution=dist)
        assert target == "crane"


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestStep:
    def test_correct_guess_solves(self, tiny_game):
        state, _ = tiny_game.new_game(word="crane")
        new_state, pattern, done = tiny_game.step(state, "crane", "crane")
        assert pattern == PATTERN_SOLVED
        assert done
        assert new_state.solved

    def test_wrong_guess_narrows_candidates(self, tiny_game):
        state, _ = tiny_game.new_game(word="crane")
        new_state, _, done = tiny_game.step(state, "slate", "crane")
        assert new_state.remaining < state.remaining
        assert not done

    def test_step_after_done_raises(self, tiny_game):
        state, _ = tiny_game.new_game(word="crane")
        state, _, _ = tiny_game.step(state, "crane", "crane")
        with pytest.raises(ValueError):
            tiny_game.step(state, "slate", "crane")

    def test_invalid_word_raises(self, tiny_game):
        state, _ = tiny_game.new_game(word="crane")
        with pytest.raises(ValueError):
            tiny_game.step(state, "xyzzy", "crane")

    def test_candidates_consistent_with_pattern(self, tiny_game):
        """All remaining candidates must produce the same pattern against the guess."""
        from wordle.pattern import compute_pattern
        state, _ = tiny_game.new_game(word="groan")
        new_state, pattern, _ = tiny_game.step(state, "slate", "groan")
        for ci in new_state.candidates:
            word = tiny_game.words[ci]
            assert compute_pattern("slate", word) == pattern


# ---------------------------------------------------------------------------
# Full game loop
# ---------------------------------------------------------------------------

class TestFullLoop:
    def test_random_policy_terminates(self, tiny_game):
        policy = RandomPolicy(rng=np.random.default_rng(1))
        for word in tiny_game.words:
            state, target = tiny_game.new_game(word=word)
            turns = 0
            while not state.done:
                guess = policy(state, tiny_game)
                state, _, _ = tiny_game.step(state, guess, target)
                turns += 1
            assert turns <= tiny_game.max_guesses

    def test_always_solvable_if_target_in_candidates(self, tiny_game):
        """
        The target is always in the candidate set until it is guessed,
        because candidates are filtered to words consistent with all patterns.
        """
        for word in tiny_game.words:
            state, target = tiny_game.new_game(word=word)
            target_idx = tiny_game._word_to_idx[target]
            while not state.done:
                assert target_idx in state.candidates
                state, _, _ = tiny_game.step(state, target, target)
            assert state.solved
