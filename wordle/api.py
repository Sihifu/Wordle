"""FastAPI web interface for the Wordle solver.

Run with:
    uvicorn wordle.api:app --host 127.0.0.1 --port 8000

Or directly:
    python -m wordle.api
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .game import GameState, WordleGame
from .policy import entropy as _entropy


_STATIC = Path(__file__).parent / "static"
_game: WordleGame | None = None
_current_min_zipf: float = 1.0
_current_lang: str = 'en'
# Caches keyed by (min_zipf, lang) so English and German vocabs coexist.
_game_cache: dict[tuple, WordleGame] = {}
_initial_cache: dict[tuple, dict] = {}


def _make_initial_response(game: WordleGame) -> dict:
    """Compute the opening suggestion (all candidates, no history). Expensive for large vocabs."""
    state = GameState(candidates=frozenset(range(len(game.words))), history=())
    best_gi, ent, is_cand, top, cur_ent = _analyse(game, state)
    return {
        "remaining": len(game.words),
        "solved": False,
        "failed": False,
        "guess_count": 0,
        "current_entropy": round(cur_ent, 2),
        "suggestion": game.words[best_gi],
        "suggestion_entropy": round(float(ent[best_gi]), 2),
        "top": top,
    }


_PRELOAD_ZIPF_LEVELS = [1.2, 0.7, 0.2, 0.0]  # preloaded for every language below
_PRELOAD_LANGS = ['en', 'de']


def _preload_all() -> None:
    """Background thread: build all vocab sizes for each preloaded language, smallest first."""
    if os.getenv("WORDLE_SKIP_PRELOAD"):
        return
    for lang in _PRELOAD_LANGS:
        for zipf in _PRELOAD_ZIPF_LEVELS:
            key = (zipf, lang)
            if key in _game_cache:
                continue
            try:
                print(f"[preload] min_zipf={zipf} lang={lang} loading…", flush=True)
                game = WordleGame.build(min_zipf=zipf, lang=lang)
                _game_cache[key] = game
                _initial_cache[key] = _make_initial_response(game)
                print(f"[preload] min_zipf={zipf} lang={lang} ready ({len(game.words):,} words)", flush=True)
            except Exception as exc:
                print(f"[preload] min_zipf={zipf} lang={lang} failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _game
    # Load default vocab synchronously so the server is immediately usable.
    _game = WordleGame.build()
    cache_key = (_current_min_zipf, _current_lang)
    _game_cache[cache_key] = _game
    _initial_cache[cache_key] = _make_initial_response(_game)
    # Pre-load all other English vocab sizes in the background.
    threading.Thread(target=_preload_all, daemon=True, name="vocab-preload").start()
    yield


app = FastAPI(title="Wordle Solver", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def root():
    return FileResponse(
        _STATIC / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GuessEntry(BaseModel):
    guess: str
    pattern: list[int]  # 5 ints: 0=grey, 1=yellow, 2=green


class SolveRequest(BaseModel):
    history: list[GuessEntry]
    candidates_only: bool = True   # default: only suggest possible answers


class ConfigRequest(BaseModel):
    min_zipf: float
    lang: str = 'en'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode(digits: list[int]) -> int:
    result = 0
    for d in digits:
        result = result * 3 + d
    return result


def _prior(game: WordleGame, cands: np.ndarray) -> np.ndarray:
    probs = np.array([game.pm.distribution.probability(game.words[c]) for c in cands])
    s = probs.sum()
    return np.ones(len(cands)) / len(cands) if s == 0 else probs / s


def _all_entropies(
    game: WordleGame,
    guess_idx: np.ndarray,
    cands: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Expected information (bits) for each word in guess_idx against cands.

    guess_idx may be a subset of the vocabulary (e.g. only candidates).
    Chunked to keep peak working set ~8 MB instead of the full n² allocation.
    Returns shape (len(guess_idx),).
    """
    n_g = len(guess_idx)
    n_c = len(cands)
    chunk = max(32, min(n_g, 8 * 1024 * 1024 // max(1, n_c * 4)))
    entropies = np.empty(n_g, dtype=np.float64)

    for start in range(0, n_g, chunk):
        end = min(start + chunk, n_g)
        size = end - start
        sub = game.pm.matrix[guess_idx[start:end], :][:, cands].astype(np.int32)
        offsets = np.arange(size, dtype=np.int32)[:, np.newaxis] * 243
        flat = (sub + offsets).ravel()
        weights = np.tile(prior, size)
        marg = np.bincount(flat, weights=weights, minlength=size * 243).reshape(size, 243)
        p = marg
        log_p = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
        entropies[start:end] = -np.sum(p * log_p, axis=1)

    return entropies


def _reconstruct(game: WordleGame, history: list[GuessEntry]) -> GameState:
    candidates: frozenset = frozenset(range(len(game.words)))
    h: tuple = ()
    for entry in history:
        word = entry.guess.lower().strip()
        pattern = _encode(entry.pattern)
        gi = game._word_to_idx.get(word)
        if gi is not None:
            # Fast path: pattern already in the matrix.
            new_candidates = frozenset(
                c for c in candidates if game.pm.matrix[gi, c] == pattern
            )
        else:
            # Guess not in this vocabulary — compute patterns on-the-fly so the
            # constraint is still applied (important when replaying history after
            # a vocabulary switch that drops the guess word).
            from wordle.pattern import compute_pattern
            new_candidates = frozenset(
                c for c in candidates
                if compute_pattern(word, game.words[c]) == pattern
            )
        candidates = new_candidates
        h = h + ((word, pattern),)
    return GameState(candidates=candidates, history=h)


def _analyse(
    game: WordleGame,
    state: GameState,
    top_k: int = 5,
    candidates_only: bool = False,
):
    """Return (best_gi, full_ent, is_candidate_mask, top_k_list, current_entropy).

    candidates_only=True  → only candidate words are considered as guesses (fast,
                            never suggests non-answer words).
    candidates_only=False → full vocabulary evaluated (may surface non-candidate
                            words with higher expected info).
    full_ent is always shape (n_words,); non-evaluated words have entropy 0.
    """
    cands = np.array(sorted(state.candidates))
    prior = _prior(game, cands)

    is_cand = np.zeros(len(game.words), dtype=bool)
    is_cand[cands] = True

    guess_idx = cands if candidates_only else np.arange(len(game.words))
    ent_local = _all_entropies(game, guess_idx, cands, prior)  # shape (len(guess_idx),)

    # Scores within the evaluated set; tie-break: prefer candidates.
    scores_local = ent_local + is_cand[guess_idx] * 1e-9
    best_local = int(np.argmax(scores_local))
    best_gi = int(guess_idx[best_local])

    # Stable descending sort: equal scores preserve original (alphabetical) order,
    # matching np.argmax which always returns the first maximum.
    sorted_local = np.argsort(-scores_local, kind='stable')
    gi_to_local = {int(guess_idx[li]): li for li in range(len(guess_idx))}
    filtered = [int(guess_idx[li]) for li in sorted_local
                if ent_local[li] > 0.01 or is_cand[guess_idx[li]]][:top_k]
    if not filtered:
        filtered = [best_gi]

    top = [
        {
            "word": game.words[gi],
            "entropy": round(float(ent_local[gi_to_local[gi]]), 2),
            "is_candidate": bool(is_cand[gi]),
        }
        for gi in filtered
    ]

    # Build full entropy array (zeros for non-evaluated words).
    full_ent = np.zeros(len(game.words))
    full_ent[guess_idx] = ent_local

    return best_gi, full_ent, is_cand, top, float(_entropy(prior))


def _guess_entropy(
    word: str,
    gi: int | None,
    cands: np.ndarray,
    prior: np.ndarray,
    game: WordleGame,
) -> float:
    """
    Expected information (bits) for guessing *word* against *cands*.

    If *gi* is the word's index in the current vocabulary the pre-computed matrix
    is used (O(n_cands) lookup).  If *gi* is None the word is not in the
    vocabulary and patterns are computed on-the-fly with compute_pattern — still
    O(n_cands), just slightly slower due to Python-level pattern computation.
    """
    if gi is not None:
        patterns = game.pm.matrix[gi, cands].astype(np.int64)
    else:
        from wordle.pattern import compute_pattern
        patterns = np.array(
            [compute_pattern(word, game.words[c]) for c in cands], dtype=np.int64
        )
    marginal = np.bincount(patterns, weights=prior, minlength=243).astype(np.float64)
    p = marginal[marginal > 0]
    return float(-np.dot(p, np.log2(p))) if len(p) else 0.0


def _simulate(game: WordleGame, answer: str, candidates_only: bool = False) -> list[dict]:
    """Run the solver against a known answer; return the play trace with entropy.

    candidates_only=True  → solver only picks from remaining candidate words (Answers only mode).
    candidates_only=False → solver may pick any word in the vocabulary (All words mode).
    """
    state, _ = game.new_game(word=answer)
    trace = []
    while not state.done:
        cands = np.array(sorted(state.candidates))
        prior = _prior(game, cands)

        # Use _analyse so the mode is respected consistently with the live solver.
        best_gi, _, _, _, _ = _analyse(game, state, top_k=1, candidates_only=candidates_only)
        guess = game.words[best_gi]
        guess_gi = game._word_to_idx[guess]
        entropy_gained = _guess_entropy(guess, guess_gi, cands, prior, game)

        state, pattern, _ = game.step(state, guess, answer)

        digits: list[int] = []
        p = pattern
        for _ in range(5):
            digits.append(p % 3)
            p //= 3

        # Actual entropy remaining in the posterior after this guess.
        if state.done:
            entropy_after = 0.0
        else:
            post_cands = np.array(sorted(state.candidates))
            post_prior = _prior(game, post_cands)
            entropy_after = float(_entropy(post_prior))

        trace.append({
            "guess": guess,
            "pattern": list(reversed(digits)),
            "entropy_gained": round(entropy_gained, 2),
            "entropy_after": round(entropy_after, 2),
        })
    return trace


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/api/solve")
def solve(req: SolveRequest):
    game = _game
    cache_key = (_current_min_zipf, _current_lang)
    # Fast path: opening state with no history has been pre-computed.
    if not req.history and cache_key in _initial_cache:
        return _initial_cache[cache_key]
    cur = _reconstruct(game, req.history)

    resp: dict = {
        "remaining": cur.remaining,
        "solved": cur.solved,
        "failed": cur.failed,
        "guess_count": cur.guess_count,
    }

    # Compare last user guess against solver's optimal — always, including on the
    # final turn, so chart bars and history annotation are never missing.
    # Works even when the user's guess is not in the current vocabulary.
    if req.history:
        pre = _reconstruct(game, req.history[:-1])
        last_word = req.history[-1].guess.lower().strip()
        if pre.remaining > 0:
            gi = game._word_to_idx.get(last_word)
            pre_best_gi, pre_ent, pre_is_cand, _, _ = _analyse(
                game, pre, candidates_only=req.candidates_only
            )
            pre_cands = np.array(sorted(pre.candidates))
            pre_prior = _prior(game, pre_cands)
            your_ent = _guess_entropy(last_word, gi, pre_cands, pre_prior, game)
            resp["comparison"] = {
                "your_guess": last_word,
                "your_entropy": round(your_ent, 2),
                "your_in_vocab": gi is not None,
                "solver_guess": game.words[pre_best_gi],
                "solver_entropy": round(float(pre_ent[pre_best_gi]), 2),
                "solver_is_candidate": bool(pre_is_cand[pre_best_gi]),
                "delta": round(float(pre_ent[pre_best_gi]) - your_ent, 2),
            }

    # Solved: simulate both solver modes so the frontend can always show both paths.
    if cur.solved and req.history:
        answer = req.history[-1].guess.lower().strip()
        if answer in game._word_to_idx:
            trace_ans = _simulate(game, answer, candidates_only=True)
            trace_all = _simulate(game, answer, candidates_only=False)
            resp["solver_trace_answers"]   = trace_ans
            resp["solver_trace_all"]       = trace_all
            resp["solver_guesses_answers"] = len(trace_ans)
            resp["solver_guesses_all"]     = len(trace_all)
            resp["initial_entropy"] = _initial_cache.get(cache_key, {}).get("current_entropy")
        return resp

    if cur.failed or cur.remaining == 0:
        return resp

    # Ongoing game: add suggestion and stats for the current state.
    best_gi, ent, is_cand, top, cur_ent = _analyse(
        game, cur, candidates_only=req.candidates_only
    )
    resp.update({
        "current_entropy": round(cur_ent, 2),
        "suggestion": game.words[best_gi],
        "suggestion_entropy": round(float(ent[best_gi]), 2),
        "top": top,
    })

    return resp


class SimulateRequest(BaseModel):
    answer: str


@app.post("/api/simulate")
def simulate_answer(req: SimulateRequest):
    """Return both solver traces for a given answer (used by the manual-solve UI).
    Works even when the user reached game-over without solving."""
    game = _game
    answer = req.answer.lower().strip()
    if answer not in game._word_to_idx:
        return {"in_vocab": False}
    trace_ans = _simulate(game, answer, candidates_only=True)
    trace_all = _simulate(game, answer, candidates_only=False)
    cache_key = (_current_min_zipf, _current_lang)
    return {
        "in_vocab": True,
        "solver_trace_answers":   trace_ans,
        "solver_trace_all":       trace_all,
        "solver_guesses_answers": len(trace_ans),
        "solver_guesses_all":     len(trace_all),
        "initial_entropy":        _initial_cache.get(cache_key, {}).get("current_entropy"),
    }


@app.get("/api/status")
def get_status():
    """Return current vocabulary metadata."""
    return {"word_count": len(_game.words), "min_zipf": _current_min_zipf, "lang": _current_lang}


@app.post("/api/config")
def configure(req: ConfigRequest):
    """Switch vocabulary and/or language.  Loads game from disk on first use, then stays in memory.
    Also pre-computes the opening suggestion so the first /api/solve is instant."""
    global _game, _current_min_zipf, _current_lang
    min_zipf = max(0.0, min(2.0, req.min_zipf))
    lang = req.lang.strip().lower() or 'en'
    cache_key = (min_zipf, lang)
    in_memory = cache_key in _game_cache
    if not in_memory:
        _game_cache[cache_key] = WordleGame.build(min_zipf=min_zipf, lang=lang)
    if cache_key not in _initial_cache:
        _initial_cache[cache_key] = _make_initial_response(_game_cache[cache_key])
    _game = _game_cache[cache_key]
    _current_min_zipf = min_zipf
    _current_lang = lang
    return {"word_count": len(_game.words), "min_zipf": min_zipf, "lang": lang, "in_memory": in_memory}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wordle.api:app", host="127.0.0.1", port=8000, reload=False)
