"""Tests for WordDistribution vocabulary and probability distribution."""

import pytest
import numpy as np
from wordle.words import WordDistribution


WORDS = ["crane", "slate", "audio", "stale", "groan"]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_list_uniform(self):
        d = WordDistribution(WORDS)
        p = d.probability("crane")
        assert pytest.approx(p) == 1 / len(WORDS)

    def test_from_dict(self):
        d = WordDistribution({"crane": 3.0, "slate": 1.0})
        assert pytest.approx(d.probability("crane")) == 0.75
        assert pytest.approx(d.probability("slate")) == 0.25

    def test_from_list_zipf(self):
        d = WordDistribution(WORDS, zipf=True)
        total = sum(d.probability(w) for w in WORDS)
        assert pytest.approx(total) == 1.0
        # common word "audio" should outweigh rare "groan"
        assert d.probability("audio") > d.probability("groan")

    def test_words_sorted_alphabetically(self):
        d = WordDistribution(["slate", "crane", "audio"])
        assert d.words == sorted(["slate", "crane", "audio"])

    def test_probabilities_sum_to_one(self):
        d = WordDistribution(WORDS)
        assert pytest.approx(sum(d.probability(w) for w in WORDS)) == 1.0

    def test_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            WordDistribution([])

    def test_len(self):
        d = WordDistribution(WORDS)
        assert len(d) == len(WORDS)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def test_probability_known_word(self):
        d = WordDistribution(WORDS)
        assert d.probability("crane") > 0

    def test_probability_unknown_word(self):
        d = WordDistribution(WORDS)
        assert d.probability("zzzzz") == 0.0

    def test_contains_true(self):
        d = WordDistribution(WORDS)
        assert d.contains("slate")

    def test_contains_false(self):
        d = WordDistribution(WORDS)
        assert not d.contains("xyzzy")

    def test_word_to_idx_consistent(self):
        d = WordDistribution(WORDS)
        for i, w in enumerate(d.words):
            assert d._word_to_idx[w] == i


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_returns_known_word(self):
        d = WordDistribution(WORDS)
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert d.sample(rng) in WORDS

    def test_sample_integer_seed_reproducible(self):
        d = WordDistribution(WORDS)
        assert d.sample(42) == d.sample(42)

    def test_sample_biased_toward_high_weight(self):
        d = WordDistribution({"crane": 1000.0, "slate": 1.0})
        rng = np.random.default_rng(0)
        samples = [d.sample(rng) for _ in range(200)]
        assert samples.count("crane") > 190


# ---------------------------------------------------------------------------
# Corpus constructor (smoke test — just checks shape and normalisation)
# ---------------------------------------------------------------------------

class TestFromWordfreq:
    def test_returns_nonempty(self):
        d = WordDistribution.default()
        assert len(d) > 100

    def test_probabilities_normalised(self):
        d = WordDistribution.default()
        total = sum(d._weights)
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_all_five_letter_lowercase(self):
        d = WordDistribution.default()
        for w in d.words:
            assert len(w) == 5 and w.isalpha() and w.islower()


class TestFromWordfreqLang:
    def test_german_returns_nonempty(self):
        d = WordDistribution.from_wordfreq(min_zipf=1.5, lang="de")
        assert len(d) > 50

    def test_german_words_five_letter_lowercase(self):
        d = WordDistribution.from_wordfreq(min_zipf=1.5, lang="de")
        for w in d.words[:30]:
            assert len(w) == 5 and w.isalpha() and w.islower()

    def test_german_probabilities_normalised(self):
        d = WordDistribution.from_wordfreq(min_zipf=1.5, lang="de")
        assert pytest.approx(sum(d._weights), abs=1e-6) == 1.0

    def test_english_is_default_lang(self):
        d_default = WordDistribution.from_wordfreq(min_zipf=1.5)
        d_en = WordDistribution.from_wordfreq(min_zipf=1.5, lang="en")
        assert d_default.words == d_en.words

    def test_english_and_german_vocabularies_differ(self):
        en = WordDistribution.from_wordfreq(min_zipf=1.5, lang="en")
        de = WordDistribution.from_wordfreq(min_zipf=1.5, lang="de")
        assert set(en.words) != set(de.words)
