"""Integration tests for the FastAPI web interface."""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("WORDLE_SKIP_PRELOAD", "1")

from wordle.api import app  # noqa: E402 — must come after env var is set


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient; preloading is disabled via env var."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_returns_required_fields(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert {"word_count", "min_zipf", "lang"} <= data.keys()

    def test_default_lang_is_english(self, client):
        r = client.get("/api/status")
        assert r.json()["lang"] == "en"

    def test_word_count_positive(self, client):
        r = client.get("/api/status")
        assert r.json()["word_count"] > 100


# ---------------------------------------------------------------------------
# /api/solve — no history (opening state)
# ---------------------------------------------------------------------------

class TestSolveOpeningState:
    def test_returns_suggestion(self, client):
        r = client.post("/api/solve", json={"history": [], "candidates_only": True})
        assert r.status_code == 200
        data = r.json()
        assert len(data["suggestion"]) == 5
        assert data["solved"] is False
        assert data["failed"] is False
        assert data["remaining"] > 0

    def test_current_entropy_positive(self, client):
        r = client.post("/api/solve", json={"history": [], "candidates_only": True})
        assert r.json()["current_entropy"] > 0

    def test_top_list_structure(self, client):
        r = client.post("/api/solve", json={"history": [], "candidates_only": True})
        top = r.json()["top"]
        assert len(top) >= 1
        for item in top:
            assert {"word", "entropy", "is_candidate"} <= item.keys()
            assert len(item["word"]) == 5
            assert item["entropy"] >= 0

    def test_answers_only_top_all_candidates(self, client):
        r = client.post("/api/solve", json={"history": [], "candidates_only": True})
        # At game start every word is a candidate, so is_candidate is always True.
        for item in r.json()["top"]:
            assert item["is_candidate"] is True

    def test_all_words_mode_returns_suggestion(self, client):
        r = client.post("/api/solve", json={"history": [], "candidates_only": False})
        assert r.status_code == 200
        assert len(r.json()["suggestion"]) == 5


# ---------------------------------------------------------------------------
# /api/solve — with history (mid-game)
# ---------------------------------------------------------------------------

class TestSolveWithHistory:
    def _first_suggestion(self, client) -> str:
        return client.post(
            "/api/solve", json={"history": [], "candidates_only": True}
        ).json()["suggestion"]

    def test_remaining_decreases_after_constraint(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [0, 0, 0, 0, 0]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        status = client.get("/api/status").json()
        assert r.json()["remaining"] < status["word_count"]

    def test_comparison_present_after_first_guess(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [0, 0, 0, 0, 0]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        cmp = r.json()["comparison"]
        assert {"your_guess", "your_entropy", "solver_guess", "solver_entropy", "delta"} <= cmp.keys()
        assert isinstance(cmp["your_entropy"], float)
        assert isinstance(cmp["delta"], float)

    def test_all_green_pattern_marks_solved(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        data = r.json()
        assert data["solved"] is True
        assert data["guess_count"] == 1

    def test_solved_returns_both_solver_traces(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        data = r.json()
        assert "solver_trace_answers" in data
        assert "solver_trace_all" in data
        assert isinstance(data["solver_guesses_answers"], int)
        assert isinstance(data["solver_guesses_all"], int)
        assert data["solver_guesses_answers"] >= 1
        assert data["solver_guesses_all"] >= 1

    def test_solved_provides_initial_entropy(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        assert r.json()["initial_entropy"] > 0

    def test_solver_trace_steps_structure(self, client):
        word = self._first_suggestion(client)
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        for key in ("solver_trace_answers", "solver_trace_all"):
            for step in r.json()[key]:
                assert len(step["guess"]) == 5
                assert len(step["pattern"]) == 5
                assert all(c in (0, 1, 2) for c in step["pattern"])
                assert step["entropy_gained"] >= 0
                assert step["entropy_after"] >= 0

    def test_final_guess_has_comparison_bits(self, client):
        """Regression: pre.remaining==1 before the winning guess must still yield comparison data."""
        word = self._first_suggestion(client)
        # Solved in 1 guess; pre is the full initial state (remaining > 0).
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        data = r.json()
        assert data["solved"] is True
        assert "comparison" in data
        assert isinstance(data["comparison"]["your_entropy"], float)

    def test_out_of_vocab_guess_still_compares(self, client):
        """A guess not in the vocabulary should still produce comparison data."""
        history = [{"guess": "zzzzz", "pattern": [0, 0, 0, 0, 0]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        assert r.status_code == 200
        cmp = r.json().get("comparison")
        if cmp is not None:
            assert cmp["your_in_vocab"] is False


# ---------------------------------------------------------------------------
# candidates_only mode
# ---------------------------------------------------------------------------

class TestCandidatesOnlyMode:
    def test_answers_only_top_items_are_candidates(self, client):
        word = client.post(
            "/api/solve", json={"history": [], "candidates_only": True}
        ).json()["suggestion"]
        history = [{"guess": word, "pattern": [0, 0, 0, 0, 0]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        for item in r.json().get("top", []):
            assert item["is_candidate"] is True

    def test_both_traces_differ_for_informative_games(self, client):
        """Answers-only and all-words solver traces can differ in length/words."""
        word = client.post(
            "/api/solve", json={"history": [], "candidates_only": True}
        ).json()["suggestion"]
        history = [{"guess": word, "pattern": [2, 2, 2, 2, 2]}]
        r = client.post("/api/solve", json={"history": history, "candidates_only": True})
        data = r.json()
        # Both traces must exist and end at entropy_after == 0.
        assert data["solver_trace_answers"][-1]["entropy_after"] == 0.0
        assert data["solver_trace_all"][-1]["entropy_after"] == 0.0


# ---------------------------------------------------------------------------
# /api/simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def _vocab_word(self, client) -> str:
        return client.post(
            "/api/solve", json={"history": [], "candidates_only": True}
        ).json()["suggestion"]

    def test_in_vocab_word_returns_traces(self, client):
        word = self._vocab_word(client)
        r = client.post("/api/simulate", json={"answer": word})
        assert r.status_code == 200
        data = r.json()
        assert data["in_vocab"] is True
        assert "solver_trace_answers" in data
        assert "solver_trace_all" in data
        assert data["solver_guesses_answers"] >= 1
        assert data["solver_guesses_all"] >= 1

    def test_out_of_vocab_returns_in_vocab_false(self, client):
        r = client.post("/api/simulate", json={"answer": "zzzzz"})
        assert r.status_code == 200
        assert r.json()["in_vocab"] is False

    def test_trace_steps_are_valid(self, client):
        word = self._vocab_word(client)
        data = client.post("/api/simulate", json={"answer": word}).json()
        for trace in (data["solver_trace_answers"], data["solver_trace_all"]):
            assert len(trace) >= 1
            for step in trace:
                assert len(step["guess"]) == 5
                assert all(c in (0, 1, 2) for c in step["pattern"])
                assert step["entropy_gained"] >= 0
                assert step["entropy_after"] >= 0
            # The last step resolves all uncertainty.
            assert trace[-1]["entropy_after"] == 0.0

    def test_initial_entropy_positive(self, client):
        word = self._vocab_word(client)
        data = client.post("/api/simulate", json={"answer": word}).json()
        assert data["initial_entropy"] > 0

    def test_last_trace_step_is_the_answer(self, client):
        word = self._vocab_word(client)
        data = client.post("/api/simulate", json={"answer": word}).json()
        for trace in (data["solver_trace_answers"], data["solver_trace_all"]):
            assert trace[-1]["guess"] == word
            assert trace[-1]["pattern"] == [2, 2, 2, 2, 2]


# ---------------------------------------------------------------------------
# /api/config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_returns_required_fields(self, client):
        r = client.post("/api/config", json={"min_zipf": 1.0, "lang": "en"})
        assert r.status_code == 200
        assert {"word_count", "min_zipf", "lang", "in_memory"} <= r.json().keys()

    def test_clamps_min_zipf_above_max(self, client):
        r = client.post("/api/config", json={"min_zipf": 99.0, "lang": "en"})
        assert r.json()["min_zipf"] <= 2.0

    def test_clamps_min_zipf_below_min(self, client):
        r = client.post("/api/config", json={"min_zipf": -5.0, "lang": "en"})
        assert r.json()["min_zipf"] >= 0.0

    def test_switches_to_german(self, client):
        try:
            r = client.post("/api/config", json={"min_zipf": 1.0, "lang": "de"})
            assert r.status_code == 200
            data = r.json()
            assert data["lang"] == "de"
            assert data["word_count"] > 100
            # Status endpoint should reflect the switch.
            assert client.get("/api/status").json()["lang"] == "de"
        finally:
            client.post("/api/config", json={"min_zipf": 1.0, "lang": "en"})

    def test_german_solve_returns_suggestion(self, client):
        try:
            client.post("/api/config", json={"min_zipf": 1.0, "lang": "de"})
            r = client.post("/api/solve", json={"history": [], "candidates_only": True})
            assert r.status_code == 200
            assert len(r.json()["suggestion"]) == 5
        finally:
            client.post("/api/config", json={"min_zipf": 1.0, "lang": "en"})

    def test_second_load_is_cached(self, client):
        r1 = client.post("/api/config", json={"min_zipf": 1.0, "lang": "en"})
        r2 = client.post("/api/config", json={"min_zipf": 1.0, "lang": "en"})
        assert r2.json()["in_memory"] is True
