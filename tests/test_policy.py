"""Tests for policy utility functions and EntropyPolicy."""

import pytest
import numpy as np
from wordle.pattern import PATTERN_SOLVED, compute_pattern
from wordle.policy import (
    EntropyPolicy,
    pattern_marginal,
    entropy,
    bayesian_update,
)


WORDS = ["crane", "slate", "audio", "stale", "groan"]


# ---------------------------------------------------------------------------
# pattern_marginal
# ---------------------------------------------------------------------------

class TestPatternMarginal:
    def test_shape(self, tiny_game):
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["crane"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        marg = pattern_marginal(gi, candidates, prior, pm)
        assert marg.shape == (243,)

    def test_sums_to_one(self, tiny_game):
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["slate"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        marg = pattern_marginal(gi, candidates, prior, pm)
        assert pytest.approx(marg.sum()) == 1.0

    def test_solved_pattern_nonzero_for_self(self, tiny_game):
        """The all-green pattern must have positive probability when the guess
        word is among the candidates (it matches itself)."""
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["audio"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        marg = pattern_marginal(gi, candidates, prior, pm)
        assert marg[PATTERN_SOLVED] > 0

    def test_single_candidate_gives_delta(self, tiny_game):
        """With one candidate, the marginal is a delta on the pattern
        of (guess, that candidate)."""
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["crane"]
        ci = np.array([pm._dist._word_to_idx["slate"]])
        prior = np.ones(1)
        marg = pattern_marginal(gi, ci, prior, pm)
        expected_pattern = compute_pattern("crane", "slate")
        assert marg[expected_pattern] == pytest.approx(1.0)
        assert marg.sum() == pytest.approx(1.0)

    def test_weighted_prior_shifts_mass(self, tiny_game):
        """Concentrating prior on one word should increase that word's
        pattern probability and decrease all others'."""
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["crane"]
        candidates = np.arange(len(WORDS))
        # Put all mass on "audio" (index 0 in sorted WORDS = audio)
        audio_idx = pm._dist._word_to_idx["audio"]
        local_idx = list(candidates).index(audio_idx)
        heavy = np.zeros(len(WORDS))
        heavy[local_idx] = 1.0
        marg = pattern_marginal(gi, candidates, heavy, pm)
        assert pytest.approx(marg.sum()) == 1.0
        # The only non-zero entry is the pattern of crane vs audio
        expected = compute_pattern("crane", "audio")
        assert marg[expected] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_is_maximum(self):
        n = 8
        uniform = np.ones(n) / n
        assert pytest.approx(entropy(uniform)) == np.log2(n)

    def test_delta_is_zero(self):
        delta = np.array([0.0, 0.0, 1.0, 0.0])
        assert pytest.approx(entropy(delta)) == 0.0

    def test_binary_half_half(self):
        p = np.array([0.5, 0.5])
        assert pytest.approx(entropy(p)) == 1.0

    def test_ignores_zero_entries(self):
        p = np.array([0.5, 0.0, 0.5])
        assert pytest.approx(entropy(p)) == 1.0

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            p = rng.dirichlet(np.ones(6))
            assert entropy(p) >= 0


# ---------------------------------------------------------------------------
# bayesian_update
# ---------------------------------------------------------------------------

class TestBayesianUpdate:
    def test_inconsistent_candidates_removed(self, tiny_game):
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["crane"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        observed = pm.get("crane", "slate")
        new_cands, new_prior = bayesian_update(prior, candidates, gi, observed, pm)
        for c in new_cands:
            assert compute_pattern("crane", tiny_game.words[c]) == observed

    def test_new_prior_sums_to_one(self, tiny_game):
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["slate"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        observed = pm.get("slate", "audio")
        _, new_prior = bayesian_update(prior, candidates, gi, observed, pm)
        assert pytest.approx(new_prior.sum()) == 1.0

    def test_solved_pattern_leaves_one_candidate(self, tiny_game):
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["groan"]
        candidates = np.arange(len(WORDS))
        prior = np.ones(len(WORDS)) / len(WORDS)
        new_cands, new_prior = bayesian_update(
            prior, candidates, gi, PATTERN_SOLVED, pm
        )
        assert len(new_cands) == 1
        assert tiny_game.words[new_cands[0]] == "groan"
        assert pytest.approx(new_prior[0]) == 1.0

    def test_weighted_prior_preserved_proportionally(self, tiny_game):
        """Words that survive the update should keep their relative weights."""
        pm = tiny_game.pm
        gi = pm._dist._word_to_idx["crane"]
        candidates = np.arange(len(WORDS))
        weights = np.array([3.0, 1.0, 2.0, 1.0, 0.5])
        prior = weights / weights.sum()
        observed = pm.get("crane", "slate")
        new_cands, new_prior = bayesian_update(prior, candidates, gi, observed, pm)
        # Survivors: PM indices whose word produces the observed pattern
        survivors = [i for i in range(len(tiny_game.words))
                     if compute_pattern("crane", tiny_game.words[i]) == observed]
        survivor_weights = weights[survivors]
        expected_prior = survivor_weights / survivor_weights.sum()
        assert np.allclose(new_prior, expected_prior)


# ---------------------------------------------------------------------------
# EntropyPolicy
# ---------------------------------------------------------------------------

class TestEntropyPolicy:
    def test_solves_every_word(self, tiny_game):
        policy = EntropyPolicy()
        for word in WORDS:
            state, target = tiny_game.new_game(word=word)
            while not state.done:
                guess = policy(state, tiny_game)
                state, _, _ = tiny_game.step(state, guess, target)
            assert state.solved, f"EntropyPolicy failed to solve '{word}'"

    def test_guess_is_valid_word(self, tiny_game):
        policy = EntropyPolicy()
        state, _ = tiny_game.new_game(word="slate")
        guess = policy(state, tiny_game)
        assert tiny_game.is_valid_word(guess)

    def test_single_candidate_returns_it(self, tiny_game):
        """When only one candidate remains the policy must guess it."""
        from wordle.game import GameState
        policy = EntropyPolicy()
        target_idx = tiny_game._word_to_idx["audio"]
        state = GameState(
            candidates=frozenset([target_idx]),
            history=(),
            max_guesses=6,
        )
        assert policy(state, tiny_game) == "audio"

    def test_prefers_candidate_on_entropy_tie(self, tiny_game):
        """With only one candidate left the returned guess is that candidate,
        not some other word with equal (zero) entropy."""
        policy = EntropyPolicy()
        for word in WORDS:
            state, target = tiny_game.new_game(word=word)
            while not state.done:
                prev_remaining = state.remaining
                guess = policy(state, tiny_game)
                state, _, _ = tiny_game.step(state, guess, target)
                if prev_remaining == 1:
                    assert guess == target

    def test_average_guesses_beats_random(self, tiny_game):
        """EntropyPolicy should use fewer guesses on average than RandomPolicy."""
        from wordle.policy import RandomPolicy
        entropy_policy = EntropyPolicy()
        random_policy = RandomPolicy(rng=np.random.default_rng(0))

        def total_guesses(policy):
            total = 0
            for word in WORDS:
                state, target = tiny_game.new_game(word=word)
                while not state.done:
                    guess = policy(state, tiny_game)
                    state, _, _ = tiny_game.step(state, guess, target)
                total += state.guess_count
            return total

        assert total_guesses(entropy_policy) <= total_guesses(random_policy)
