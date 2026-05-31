"""Tests for GameState immutability, transitions, and derived properties."""

import pytest
from wordle.game import GameState
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


class TestLastGuessAndPattern:
    def test_last_guess_none_when_no_history(self, initial_state):
        assert initial_state.last_guess() is None

    def test_last_pattern_none_when_no_history(self, initial_state):
        assert initial_state.last_pattern() is None

    def test_last_guess_after_one_step(self, initial_state, pm):
        pattern = pm.get("crane", "slate")
        state = initial_state.update("crane", pattern, pm)
        assert state.last_guess() == "crane"

    def test_last_pattern_after_one_step(self, initial_state, pm):
        pattern = pm.get("crane", "slate")
        state = initial_state.update("crane", pattern, pm)
        assert state.last_pattern() == pattern

    def test_last_guess_reflects_most_recent(self, initial_state, pm):
        p1 = pm.get("crane", "slate")
        p2 = pm.get("slate", "slate")
        state = initial_state.update("crane", p1, pm).update("slate", p2, pm)
        assert state.last_guess() == "slate"

    def test_last_pattern_reflects_most_recent(self, initial_state, pm):
        p1 = pm.get("crane", "slate")
        p2 = pm.get("slate", "slate")
        state = initial_state.update("crane", p1, pm).update("slate", p2, pm)
        assert state.last_pattern() == p2


class TestShow:
    def test_show_runs_without_error(self, initial_state, pm):
        initial_state.show()

    def test_show_after_guesses(self, initial_state, pm):
        pattern = pm.get("crane", "slate")
        state = initial_state.update("crane", pattern, pm)
        state.show()

    def test_show_solved_state(self, initial_state, pm):
        state = initial_state.update("crane", PATTERN_SOLVED, pm)
        state.show()

    def test_show_failed_state(self, pm):
        state = GameState(
            candidates=frozenset(range(len(pm))),
            history=(),
            max_guesses=1,
        )
        pattern = pm.get("stale", "crane")
        state = state.update("stale", pattern, pm)
        state.show()


class TestRepr:
    def test_repr_contains_key_info(self, initial_state, pm):
        pattern = pm.get("crane", "slate")
        state = initial_state.update("crane", pattern, pm)
        r = repr(state)
        assert "guesses=" in r
        assert "remaining=" in r
        assert "solved=" in r
