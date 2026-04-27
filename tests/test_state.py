"""Tests for GameState immutability, transitions, and derived properties."""

import pytest
from wordle.state import GameState
from wordle.pattern import PATTERN_SOLVED, PatternMatrix
from wordle.words import WordDistribution


WORDS = ["crane", "slate", "audio", "stale"]


@pytest.fixture(scope="module")
def pm(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    return PatternMatrix(WordDistribution(WORDS), cache_dir=cache)


@pytest.fixture
def initial_state(pm):
    return GameState(
        candidates=frozenset(range(len(pm))),
        history=(),
        max_guesses=6,
    )


class TestInitialState:
    def test_no_history(self, initial_state):
        assert initial_state.guess_count == 0
        assert initial_state.history == ()

    def test_all_candidates(self, initial_state, pm):
        assert initial_state.remaining == len(pm)

    def test_not_done(self, initial_state):
        assert not initial_state.done
        assert not initial_state.solved
        assert not initial_state.failed


class TestStateUpdate:
    def test_update_returns_new_instance(self, initial_state, pm):
        new_state = initial_state.update("crane", PATTERN_SOLVED, pm)
        assert new_state is not initial_state

    def test_immutability(self, initial_state, pm):
        pattern = pm.get("slate", "crane")
        _ = initial_state.update("slate", pattern, pm)
        assert initial_state.guess_count == 0
        assert initial_state.remaining == len(pm)

    def test_candidates_filtered(self, initial_state, pm):
        new_state = initial_state.update("crane", PATTERN_SOLVED, pm)
        assert new_state.remaining == 1

    def test_history_appended(self, initial_state, pm):
        pattern = pm.get("slate", "crane")
        new_state = initial_state.update("slate", pattern, pm)
        assert len(new_state.history) == 1
        assert new_state.history[0] == ("slate", pattern)

    def test_solved_when_all_green(self, initial_state, pm):
        new_state = initial_state.update("crane", PATTERN_SOLVED, pm)
        assert new_state.solved
        assert new_state.done


class TestFailure:
    def test_failed_after_max_guesses(self, pm):
        state = GameState(
            candidates=frozenset(range(len(pm))),
            history=(),
            max_guesses=2,
        )
        pattern = pm.get("stale", "crane")
        assert pattern != PATTERN_SOLVED
        for _ in range(2):
            state = state.update("stale", pattern, pm)
        assert state.failed
        assert state.done
        assert not state.solved


class TestHashability:
    def test_can_be_used_as_dict_key(self, initial_state):
        d = {initial_state: "value"}
        assert d[initial_state] == "value"

    def test_equal_states_same_hash(self, initial_state, pm):
        s1 = initial_state.update("crane", PATTERN_SOLVED, pm)
        s2 = initial_state.update("crane", PATTERN_SOLVED, pm)
        assert s1 == s2
        assert hash(s1) == hash(s2)
