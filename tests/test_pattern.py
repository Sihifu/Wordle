"""Tests for pattern computation and PatternMatrix."""

import pytest
import numpy as np
from wordle.pattern import compute_pattern, decode_pattern, PatternMatrix, PATTERN_SOLVED
from wordle.words import WordDistribution


# ---------------------------------------------------------------------------
# compute_pattern
# ---------------------------------------------------------------------------

class TestComputePattern:
    def test_all_green(self):
        assert compute_pattern("crane", "crane") == PATTERN_SOLVED

    def test_all_grey(self):
        assert compute_pattern("moist", "carve") == 0

    def test_all_yellow(self):
        assert compute_pattern("abcde", "eabcd") == 1 * 81 + 1 * 27 + 1 * 9 + 1 * 3 + 1

    def test_mixed(self):
        # crane vs crate: c✓ r✓ a✓ n✗ e✓ → 2 2 2 0 2 = 236
        assert compute_pattern("crane", "crate") == 2 * 81 + 2 * 27 + 2 * 9 + 0 * 3 + 2

    def test_duplicate_letter_in_guess_one_in_answer(self):
        # sleep vs creep: s,l grey; e,e,p all green
        p = compute_pattern("sleep", "creep")
        digits = []
        tmp = p
        for _ in range(5):
            digits.append(tmp % 3)
            tmp //= 3
        digits.reverse()
        assert digits == [0, 0, 2, 2, 2]

    def test_duplicate_letter_both_present(self):
        # steel vs heels: s=yellow, t=grey, e=green, e=yellow, l=yellow
        p = compute_pattern("steel", "heels")
        digits = []
        tmp = p
        for _ in range(5):
            digits.append(tmp % 3)
            tmp //= 3
        digits.reverse()
        assert digits == [1, 0, 2, 1, 1]


# ---------------------------------------------------------------------------
# decode_pattern
# ---------------------------------------------------------------------------

class TestDecodePattern:
    def test_solved_is_all_green(self):
        assert decode_pattern(PATTERN_SOLVED) == "🟩🟩🟩🟩🟩"

    def test_all_grey(self):
        assert decode_pattern(0) == "⬛⬛⬛⬛⬛"

    def test_roundtrip(self):
        result = decode_pattern(81)   # 🟨⬛⬛⬛⬛
        assert result[0] == "🟨"
        assert result[1:] == "⬛⬛⬛⬛"

    def test_length(self):
        for p in [0, 81, 121, 242]:
            assert len(decode_pattern(p)) == 5


# ---------------------------------------------------------------------------
# PatternMatrix
# ---------------------------------------------------------------------------

WORDS = ["crane", "slate", "audio", "groan", "stale"]


@pytest.fixture(scope="module")
def pm(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    return PatternMatrix(WordDistribution(WORDS), cache_dir=cache)


class TestPatternMatrixConstruction:
    def test_shape(self, pm):
        assert pm.matrix.shape == (len(WORDS), len(WORDS))

    def test_dtype(self, pm):
        assert pm.matrix.dtype == np.uint8

    def test_diagonal_solved(self, pm):
        for i in range(len(WORDS)):
            assert pm.matrix[i, i] == PATTERN_SOLVED

    def test_len(self, pm):
        assert len(pm) == len(WORDS)

    def test_cache_hit(self, pm):
        pm2 = PatternMatrix(WordDistribution(WORDS), cache_dir=pm._cache_dir)
        assert np.array_equal(pm2.matrix, pm.matrix)


class TestPatternMatrixLookup:
    def test_get_matches_compute_pattern(self, pm):
        for g in WORDS:
            for a in WORDS:
                assert pm.get(g, a) == compute_pattern(g, a)

    def test_contains_true(self, pm):
        assert "crane" in pm

    def test_contains_false(self, pm):
        assert "xyzzy" not in pm


