"""Shared pytest fixtures."""

import pytest
from wordle.game import WordleGame
from wordle.pattern import PatternMatrix
from wordle.words import WordDistribution

TINY_WORDS = ["crane", "slate", "audio", "stale", "groan"]


@pytest.fixture(scope="session")
def tiny_game(tmp_path_factory):
    """5-word WordleGame used across multiple test modules."""
    cache = tmp_path_factory.mktemp("cache")
    pm = PatternMatrix(WordDistribution(TINY_WORDS), cache_dir=cache)
    return WordleGame(pm, max_guesses=6)
