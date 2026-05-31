# Wordle Solver

A mathematically rigorous Wordle solver built as a **reinforcement-learning-style agent** framework. The solver uses greedy Shannon-entropy maximisation with a Bayesian prior over word frequencies, and ships with an interactive browser UI for real-life play assistance.

---

## Mathematical Framework

### The Pattern Function

For a guess $g$ and a hidden answer $w$, define the **pattern function**

$$\varphi : \mathcal{G} \times \mathcal{A} \to \{0,1,2\}^{5}$$

where position $i$ receives:

$$\varphi(g,w)_i = \begin{cases} 2 & g_i = w_i \quad \text{(🟩 green)} \\ 1 & g_i \neq w_i,\ g_i \in \text{pool}(w,g,i) \quad \text{(🟨 yellow)} \\ 0 & \text{otherwise} \quad \text{(⬛ grey)} \end{cases}$$

The *pool* function handles duplicate letters correctly via a two-pass algorithm: greens are marked first and consumed from the pool, then yellows are matched from what remains. The pattern is encoded as a base-3 integer in $[0, 242]$; $\varphi(g,g) = 242$ (all-green) always.

### Information State

The agent maintains a **belief state** $(S, P)$ where:

- $S \subseteq \mathcal{A}$ — the set of candidate answers consistent with all observations so far
- $P : S \to [0,1]$ — a posterior probability distribution over remaining candidates, $\sum_{w \in S} P(w) = 1$

At game start, $S = \mathcal{A}$ and $P$ equals the initial prior $P_0$ (corpus-frequency-weighted or uniform).

### Marginal Distribution over Patterns

For a guess $g$, the **marginal distribution** over feedback patterns is:

$$P(f \mid g) = \sum_{w \in S} P(w) \cdot \mathbf{1}\bigl[\varphi(g,w) = f\bigr]$$

### Bayesian Update

After observing pattern $f^{\ast}$ for guess $g$:

$$P(w \mid f^{\ast}, g) = \frac{P(w) \cdot \mathbf{1}\bigl[\varphi(g,w) = f^{\ast}\bigr]}{P(f^{\ast} \mid g)}$$

Surviving candidate set: $S' = \{w \in S : \varphi(g,w) = f^{\ast}\}$.

### Entropy Maximisation (Greedy)

$$H(g, S) = -\sum_{f} P(f \mid g) \log_2 P(f \mid g)$$

At each step choose $g^{\ast} = \underset{g \in \mathcal{G}}{\text{arg max}}\ H(g, S)$.  Ties broken by preferring words still in $S$.

**Complexity:** $O(|\mathcal{G}| \cdot |S|)$ per step, fully vectorised with NumPy, chunked to stay within an ~8 MB working set.

**Note:** Locally optimal (greedy), not globally optimal. Achieves ~3.5 average guesses on a 3 500-word vocabulary.

---

## Architecture

```
wordle/
├── words.py      WordDistribution   — vocabulary + prior P₀(w), multi-language
├── pattern.py    PatternMatrix      — precomputed φ(g,w) with disk cache
├── game.py       GameState          — immutable belief state (S, history)
│               WordleGame         — simulator; new_game() / step() interface
├── policy.py     Policy ABC         — agent interface
│               RandomPolicy       — uniform random from candidates
│               HumanPolicy        — interactive stdin
│               EntropyPolicy      — greedy H(g,S) maximisation
│               pattern_marginal   — P(f|g)  [utility]
│               entropy            — H(p)     [utility]
│               bayesian_update    — Bayes posterior update
└── api.py        FastAPI server     — stateless JSON API + browser UI
```

### Data Flow

```
WordDistribution(P₀, lang)
       │
       ▼
PatternMatrix(dist)  ──── cached to data/patterns_<hash>.npy
       │
       ▼
WordleGame(pm)
       │
  new_game() ──► GameState(S=A, history=())
       │
  ┌────┴──────────────────────────────────────┐
  │  while not state.done:                    │
  │    guess  = policy(state, game)           │
  │    state, pattern, done = game.step(...)  │
  └───────────────────────────────────────────┘
```

`GameState` is **immutable** — every `update()` returns a new instance — so states can be safely hashed and used as dict keys.

---

## Installation

```bash
git clone <repo>
cd wordle
python -m venv .venv && source .venv/bin/activate

# Runtime
pip install -e .

# Development + testing
pip install -e ".[dev]"

# Web UI
pip install -e ".[web]"
```

The first call to `WordleGame.build()` computes the $n \times n$ pattern matrix and caches it to `data/`. All subsequent calls load from cache instantly.

---

## Quick Start — Python API

```python
from wordle.game import WordleGame
from wordle.policy import EntropyPolicy

game   = WordleGame.build()          # English, min_zipf=1.0 (~3 500 words)
policy = EntropyPolicy()

state, target = game.new_game()      # draw target from Zipf prior
while not state.done:
    guess         = policy(state, game)
    state, pat, _ = game.step(state, guess, target)

state.show()                         # coloured board
```

### Language support

```python
# German vocabulary (wordfreq corpus, same Zipf thresholds)
game_de = WordleGame.build(lang="de")

# Custom Zipf threshold
game = WordleGame.build(min_zipf=0.5, lang="en")  # ~8 000 English words
```

### Custom prior

```python
from wordle.words import WordDistribution

dist  = WordDistribution.from_wordfreq(lang="de")   # German, default threshold
state, target = game.new_game(word="crane")          # pin specific answer
```

---

## Web UI

A browser-based assistant for playing NYT Wordle or Süddeutsche Wordle in real time.

### Start the server

```bash
uvicorn wordle.api:app --host 127.0.0.1 --port 8000
# or
python -m wordle.api
```

Then open **http://127.0.0.1:8000** in your browser (file:// does not work — the UI needs the API).

### Features

| Feature | Description |
|---|---|
| **Live suggestions** | Best guess + top-5 list with expected information gain |
| **Two modes** | *Answers only* — solver picks from remaining candidates only; *All words* — full vocabulary search |
| **Language toggle** | Switch between English and German vocabulary mid-session |
| **Vocabulary sizes** | 5 Zipf levels (~3 k → ~11 k words) per language, all preloaded in background |
| **Performance timeline** | Chart showing entropy remaining and expected gain per turn |
| **End-of-game comparison** | Side-by-side: your path / answers-only solver / all-words solver with entropy trajectories |
| **Undo** | Step back through your guesses |
| **History annotation** | Each guess annotated as optimal / near-optimal / suboptimal vs. solver |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/solve` | Main solver — accepts `history` + `candidates_only` flag |
| `POST` | `/api/simulate` | Compute both solver traces for a given answer word |
| `POST` | `/api/config` | Switch vocabulary size (`min_zipf`) and/or language (`lang`) |
| `GET` | `/api/status` | Current vocabulary metadata |

---

## Module Reference

| Module | Key class / function | Description |
|---|---|---|
| `words.py` | `WordDistribution` | Vocabulary + $P_0(w)$. Accepts dict, list (uniform), list (Zipf). Multi-language via `lang=`. |
| `pattern.py` | `PatternMatrix` | Precomputed $\varphi(g,w)$ for all pairs. Cached to disk by word-list hash. |
| `pattern.py` | `compute_pattern(g, a)` | Pure function, returns pattern int in $[0, 242]$. |
| `pattern.py` | `decode_pattern(p)` | Returns emoji string e.g. `🟩⬛🟨🟩⬛`. |
| `game.py` | `GameState` | Frozen dataclass: `candidates`, `history`, `max_guesses`. `.show()` prints board. |
| `game.py` | `WordleGame` | Simulator. `build(min_zipf, lang)` / `new_game()` / `step()`. |
| `policy.py` | `pattern_marginal` | $P(f \mid g)$ — marginal over feedback patterns. |
| `policy.py` | `entropy` | $H(p)$ — Shannon entropy in bits. |
| `policy.py` | `bayesian_update` | Posterior update after observing a pattern. |
| `policy.py` | `EntropyPolicy` | Greedy $\text{arg max}\ H(g, S)$ with Bayesian prior. |
| `api.py` | `app` | FastAPI application. Run with `uvicorn wordle.api:app`. |

---

## Tests

```bash
pytest tests/ -v
```

126 tests covering:

- **Pattern correctness** — duplicate-letter edge cases, round-trips, matrix caching
- **GameState** — immutability, hashability, solved/failed transitions
- **WordDistribution** — construction, sampling, normalisation, multi-language
- **WordleGame** — full game loop, language switching (`en`/`de`), policy integration
- **Policy math** — `pattern_marginal`, `entropy`, `bayesian_update`, `EntropyPolicy`
- **API** — all endpoints, both solver modes, end-of-game traces, language config

---

## Notebooks

Each module has a companion Jupyter notebook in `wordle/`:

| Notebook | Demonstrates |
|---|---|
| `words_demo.ipynb` | `WordDistribution` construction, sampling, probability lookup |
| `pattern_demo.ipynb` | `PatternMatrix` build, cache |
| `game_demo.ipynb` | Full game loop with `WordleGame` |
| `policy_demo.ipynb` | All policies, marginal/entropy/Bayes utilities, aggregate stats |
