# Wordle Optimal Solver

A mathematically rigorous Wordle solver built as a **reinforcement-learning-style agent** framework. Two optimal strategies are implemented: a greedy entropy-maximising policy and (planned) an exact decision-tree policy.

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

For a guess $g$, the **marginal distribution** over feedback patterns is obtained by summing the prior over all candidates that would produce each pattern:

$$P(f \mid g) = \sum_{w \in S} P(w) \cdot \mathbf{1}\bigl[\varphi(g,w) = f\bigr]$$

With a uniform prior this reduces to $P(f \mid g) = |S_f| / |S|$, where $S_f = \{w \in S : \varphi(g,w) = f\}$.

### Bayesian Update

After observing pattern $f^{\ast}$ for guess $g$, the posterior is updated via Bayes' rule:

$$P(w \mid f^{\ast}, g) = \frac{P(w) \cdot \mathbf{1}\bigl[\varphi(g,w) = f^{\ast}\bigr]}{P(f^{\ast} \mid g)}$$

Concretely: zero out candidates inconsistent with $f^{\ast}$, then renormalise. The surviving candidate set is $S' = \{w \in S : \varphi(g,w) = f^{\ast}\}$.

---

## Approach 1 — Entropy Maximisation (Greedy)

### Entropy of a Guess

The **Shannon entropy** of guess $g$ over remaining candidates $S$ is the expected information gained:

$$H(g, S) = -\sum_{f} P(f \mid g) \log_2 P(f \mid g)$$

This measures how evenly the candidates are distributed across pattern buckets. A perfectly informative guess partitions $S$ into $|S|$ singletons, achieving $H = \log_2 |S|$ bits.

### Algorithm

At each step, choose:

$$g^{\ast} = \underset{g \in \mathcal{G}}{\text{arg max}}\ H(g, S)$$

Ties are broken by preferring words still in $S$ — they carry the same information but might already be the answer.

**Complexity:** $O(|\mathcal{G}| \cdot |S|)$ per step, fully vectorised with NumPy.

**Note:** This is locally optimal (maximises immediate information gain) but not globally optimal. It performs close to the theoretical minimum in practice (~3.5 average guesses on a 3 500-word vocabulary).

---

## Approach 2 — Optimal Decision Tree (Planned)

### Problem Formulation

Define the **total guess cost** of a strategy $\pi$ over candidates $S$:

$$C(\pi, S) = \sum_{w \in S} P(w) \cdot d_\pi(w)$$

where $d_\pi(w)$ is the number of guesses used to identify $w$ under $\pi$.

### Optimal Substructure

The optimal cost satisfies the recurrence (Bellman, 1957):

$$C^{\ast}(S) = \min_{g \in \mathcal{G}} \Bigl[ 1 + \sum_{f \neq 242} P(f \mid g) \cdot C^{\ast}(S_f) \Bigr]$$

where $S_f = \{w \in S : \varphi(g,w) = f\}$ and the sum runs over non-solved patterns only ($f = 242$ is all-green).
Base cases: $C^{\ast}(\emptyset) = 0$, $C^{\ast}(\{w\}) = 1$. The optimal expected guesses is $C^{\ast}(S_0)$.

**Complexity:** Worst case $O(2^{|\mathcal{A}|})$, tractable in practice via memoisation and branch-and-bound pruning with entropy-guided ordering.

---

## Architecture

```
wordle/
├── words.py        WordDistribution   — vocabulary + prior P₀(w)
├── pattern.py      PatternMatrix      — precomputed φ(g,w) cache
├── state.py        GameState          — immutable belief state (S, history)
├── game.py         WordleGame         — simulator (holds secret word)
└── policy.py       Policy ABC         — agent interface
                    RandomPolicy       — uniform random from candidates
                    HumanPolicy        — interactive stdin
                    EntropyPolicy      — greedy H(g,S) maximisation
                    pattern_marginal   — P(f|g)  [utility function]
                    entropy            — H(p)     [utility function]
                    bayesian_update    — Bayes posterior update
```

### Data Flow

```
WordDistribution(P₀)
       │
       ▼
PatternMatrix(dist)  ──── cached to data/patterns_<hash>.npy
       │
       ▼
WordleGame(pm)
       │
  new_game() ──► GameState(S=A, history=())
       │
  ┌────┴─────────────────────────────────────────┐
  │  while not state.done:                       │
  │    guess  = policy(state, game)              │
  │    state, pattern, done = game.step(...)     │
  └──────────────────────────────────────────────┘
```

`GameState` is **immutable** — every `update()` returns a new instance — so states can be safely hashed and used as memoisation keys for the decision-tree solver.

---

## Installation

```bash
git clone <repo>
cd wordle-solver
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first call to `WordleGame.build()` precomputes the $n \times n$ pattern matrix and caches it to `data/`. All subsequent calls load from cache instantly.

---

## Quick Start

```python
from wordle.game import WordleGame
from wordle.policy import EntropyPolicy

game   = WordleGame.build()          # loads words, builds/loads pattern matrix
policy = EntropyPolicy()

state, target = game.new_game()      # draw target from Zipf prior
while not state.done:
    guess          = policy(state, game)
    state, pat, _  = game.step(state, guess, target)

state.show()                         # coloured board
```

### Custom prior

```python
from wordle.words import WordDistribution

# Zipf-weighted: common words more likely to be the answer
dist   = WordDistribution.from_wordfreq()

# Or pin a specific word
state, target = game.new_game(word="crane")
```

### Incremental vocabulary extension

```python
# Add words without recomputing the full matrix
pm2 = game.pm.extend(["soare", "adieu"])
```

---

## Module Reference

| Module | Key class / function | Description |
|---|---|---|
| `words.py` | `WordDistribution` | Vocabulary + $P_0(w)$. Accepts dict, list (uniform), list (Zipf). |
| `pattern.py` | `PatternMatrix` | Precomputed $\varphi(g,w)$ for all pairs. Cached to disk by word-list hash. |
| `pattern.py` | `compute_pattern(g, a)` | Pure function, returns pattern int in $[0, 242]$. |
| `pattern.py` | `decode_pattern(p)` | Returns emoji string e.g. `🟩⬛🟨🟩⬛`. |
| `state.py` | `GameState` | Frozen dataclass: `candidates`, `history`, `max_guesses`. `.show()` prints board. |
| `game.py` | `WordleGame` | Simulator. `new_game()` / `step()` interface. |
| `policy.py` | `pattern_marginal` | $P(f \mid g)$ — marginal over feedback patterns. |
| `policy.py` | `entropy` | $H(p)$ — Shannon entropy in bits. |
| `policy.py` | `bayesian_update` | Posterior update after observing a pattern. |
| `policy.py` | `EntropyPolicy` | Greedy $\text{arg max}\ H(g, S)$ with Bayesian prior. |

---

## Tests

```bash
pytest tests/ -v
```

45 tests covering pattern correctness (including duplicate-letter edge cases), state immutability and hashability, and the full game loop.

---

## Notebooks

Each module has a companion Jupyter notebook in `wordle/`:

| Notebook | Demonstrates |
|---|---|
| `words_demo.ipynb` | `WordDistribution` construction, sampling, probability lookup |
| `pattern_demo.ipynb` | `PatternMatrix` build, cache, extension |
| `state_demo.ipynb` | `GameState` transitions, `show()` board rendering |
| `game_demo.ipynb` | Full game loop with `WordleGame` |
| `policy_demo.ipynb` | All policies, marginal/entropy/Bayes utilities, aggregate stats |
