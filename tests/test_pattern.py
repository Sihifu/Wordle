"""Tests for pattern computation correctness."""

import pytest
from wordle.pattern import compute_pattern, decode_pattern, build_pattern_matrix, PATTERN_SOLVED


# ---------------------------------------------------------------------------
# compute_pattern
# ---------------------------------------------------------------------------

class TestComputePattern:
    def test_all_green(self):
        assert compute_pattern("crane", "crane") == PATTERN_SOLVED

    def test_all_grey(self):
        # No letters in common
        assert compute_pattern("moist", "carve") == 0  # 00000 in base-3

    def test_all_yellow(self):
        # All letters present but all wrong positions: 11111 = 1+3+9+27+81 = 121
        assert compute_pattern("abcde", "eabcd") == 1 * 81 + 1 * 27 + 1 * 9 + 1 * 3 + 1

    def test_mixed(self):
        # "crane" vs "crate": c✓ r✓ a✓ n✗ e✓ → green green green grey green
        # = 2*81 + 2*27 + 2*9 + 0*3 + 2 = 162+54+18+0+2 = 236
        assert compute_pattern("crane", "crate") == 2 * 81 + 2 * 27 + 2 * 9 + 0 * 3 + 2

    def test_duplicate_letter_in_guess_one_in_answer(self):
        # guess "sleep", answer "creep": s,l,e,e,p vs c,r,e,e,p
        # Green pass: pos2 e==e, pos3 e==e, pos4 p==p → pool = [c,r,None,None,None]
        # Yellow pass: s not in pool → grey; l not in pool → grey
        # Result: grey grey green green green
        p = compute_pattern("sleep", "creep")
        digits = []
        tmp = p
        for _ in range(5):
            digits.append(tmp % 3)
            tmp //= 3
        digits.reverse()
        assert digits == [0, 0, 2, 2, 2]

    def test_duplicate_letter_both_present(self):
        # guess "steel", answer "heels": s,t,e,e,l vs h,e,e,l,s
        # Green pass: pos2 e==e → pool = [h,e,None,l,s]
        # Yellow pass: pos0 s→yellow (pool[4]), pos1 t→grey,
        #              pos3 e→yellow (pool[1]), pos4 l→yellow (pool[3])
        # Result: yellow grey green yellow yellow
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
        # Pattern with one yellow at pos 0: 1*81 = 81 → "🟨⬛⬛⬛⬛"
        # Each emoji is a single Unicode code point, so normal str indexing works.
        result = decode_pattern(81)
        assert result[0] == "🟨"
        assert result[1:] == "⬛⬛⬛⬛"  # positions 1-4 are grey

    def test_length(self):
        # decode_pattern always returns exactly 5 emoji characters
        import unicodedata
        for p in [0, 81, 121, 242]:
            s = decode_pattern(p)
            # Count grapheme clusters (each emoji = 1 grapheme)
            assert len(s.splitlines()[0]) == 5  # 5 emoji joined


# ---------------------------------------------------------------------------
# build_pattern_matrix
# ---------------------------------------------------------------------------

class TestBuildPatternMatrix:
    def test_shape(self):
        guesses = ["crane", "slate", "audio"]
        answers = ["crane", "slate"]
        m = build_pattern_matrix(guesses, answers)
        assert m.shape == (3, 2)

    def test_diagonal_solved(self):
        words = ["crane", "slate", "audio"]
        m = build_pattern_matrix(words, words)
        for i in range(len(words)):
            assert m[i, i] == PATTERN_SOLVED

    def test_dtype(self):
        import numpy as np
        m = build_pattern_matrix(["crane"], ["crane"])
        assert m.dtype == np.uint8

    def test_values_match_compute_pattern(self):
        guesses = ["crane", "slate"]
        answers = ["crane", "audio", "stale"]
        m = build_pattern_matrix(guesses, answers)
        for i, g in enumerate(guesses):
            for j, a in enumerate(answers):
                assert int(m[i, j]) == compute_pattern(g, a)