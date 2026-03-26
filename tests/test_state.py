"""Tests for GameState immutability, transitions, and derived properties."""

import pytest
import numpy as np
from wordle.state import GameState
from wordle.pattern import PATTERN_SOLVED, build_pattern_matrix


@pytest.fixture
def small_game():
    """A minimal 3-answer game for fast testing."""
    answers = ["crane", "slate", "audio"]
    guesses = ["crane", "slate", "audio", "stale"]
    matrix = build_pattern_matrix(guesses, answers)
    guess_idx = {g: i for i, g in enumerate(guesses)}
    answer_idx = {a: i for i, a in enumerate(answers)}
    return answers, guesses, matrix, guess_idx, answer_idx


@pytest.fixture
def initial_state(small_game):
    answers, guesses, matrix, *_ = small_game
    return GameState(
        candidates=frozenset(range(len(answers))),
        history=(),
        max_guesses=6,
    )


class TestInitialState:
    def test_no_history(self, initial_state):
        assert initial_state.guess_count == 0
        assert initial_state.history == ()

    def test_all_candidates(self, initial_state, small_game):
        answers = small_game[0]
        assert initial_state.remaining == len(answers)

    def test_not_done(self, initial_state):
        assert not initial_state.done
        assert not initial_state.solved
        assert not initial_state.failed


class TestStateUpdate:
    def test_update_returns_new_instance(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        g = "crane"
        gi = guess_idx[g]
        pattern = int(matrix[gi, answer_idx["crane"]])
        new_state = initial_state.update(g, gi, pattern, matrix)
        assert new_state is not initial_state

    def test_immutability(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        g = "slate"
        gi = guess_idx[g]
        pattern = int(matrix[gi, answer_idx["crane"]])
        _ = initial_state.update(g, gi, pattern, matrix)
        # Original state must be unchanged
        assert initial_state.guess_count == 0
        assert initial_state.remaining == 3

    def test_candidates_filtered(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        # Guess "crane" against target "crane" → PATTERN_SOLVED
        g = "crane"
        gi = guess_idx[g]
        pattern = PATTERN_SOLVED
        new_state = initial_state.update(g, gi, pattern, matrix)
        # Only "crane" itself matches all-green
        assert new_state.remaining == 1

    def test_history_appended(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        g = "slate"
        gi = guess_idx[g]
        pattern = int(matrix[gi, answer_idx["crane"]])
        new_state = initial_state.update(g, gi, pattern, matrix)
        assert len(new_state.history) == 1
        assert new_state.history[0] == (g, pattern)

    def test_solved_when_all_green(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        g = "crane"
        gi = guess_idx[g]
        new_state = initial_state.update(g, gi, PATTERN_SOLVED, matrix)
        assert new_state.solved
        assert new_state.done


class TestFailure:
    def test_failed_after_max_guesses(self, small_game):
        answers, guesses, matrix, guess_idx, answer_idx = small_game
        # Use a non-answer guess that never solves
        state = GameState(
            candidates=frozenset(range(len(answers))),
            history=(),
            max_guesses=2,
        )
        g = "stale"
        gi = guess_idx[g]
        # Two non-solving guesses
        for _ in range(2):
            pattern = int(matrix[gi, answer_idx["crane"]])
            assert pattern != PATTERN_SOLVED
            state = state.update(g, gi, pattern, matrix)

        assert state.failed
        assert state.done
        assert not state.solved


class TestHashability:
    def test_can_be_used_as_dict_key(self, initial_state):
        d = {initial_state: "value"}
        assert d[initial_state] == "value"

    def test_equal_states_same_hash(self, initial_state, small_game):
        _, guesses, matrix, guess_idx, answer_idx = small_game
        g = "crane"
        gi = guess_idx[g]
        pattern = PATTERN_SOLVED
        s1 = initial_state.update(g, gi, pattern, matrix)
        s2 = initial_state.update(g, gi, pattern, matrix)
        assert s1 == s2
        assert hash(s1) == hash(s2)
