# ChessDorrie — the Tal Troll Analyzer

A chess analysis tool that doesn't just show you the *best* move — it shows you
the **most diabolical** move: the one most likely to make a flesh-and-blood
opponent crack, hand you material, and resign in disgust.

Think Lichess analyser, but the recommendations come from Mikhail Tal after
three espressos.

## The core idea

Traditional engines (Stockfish, Lc0) optimize for **objective** outcome —
assuming the opponent finds the best reply. Humans don't.

ChessDorrie pairs an objective engine (Stockfish) with a **human-move
predictor** (Maia — a neural network trained to mimic players at a given
Elo). For every candidate move it asks:

1. How does *Stockfish* rate this move? *(objective eval)*
2. What's the distribution of replies a human at rating R is likely to play?
3. After those likely replies, what position do I end up in?
4. **Was this move a sacrifice?** Sacrifices that survive contact with a
   *human* opponent — even if Stockfish would refute them — score huge.

The output is a re-ranking: the move shown isn't necessarily the engine's
top pick. It's the move with the best **expected outcome against a real
opponent**, with extra credit for spectacle.

## What the UI shows

Per analysed position, several stacked meters:

| Meter | Meaning |
|---|---|
| **Objective eval** | Standard Stockfish centipawn evaluation. |
| **Troll eval** | Expected eval after the human's likely reply. |
| **Anger probability** | P(opponent plays a losing move) × magnitude of the loss. |
| **Human factor** | `\|troll_eval - objective_eval\|` — how much "humanness" we're exploiting. |
| **Trap depth** | Search depth at which Stockfish first finds the refutation. Deeper = more diabolical. |

Plus the standard board with multi-coloured arrows (objective best vs. troll
best vs. trap candidates) and a candidate-move table with all metrics per
move.

## Architecture

```
                       ┌────────────────────────────┐
                       │       FastAPI server       │
                       │   (app/server.py)          │
                       └────────────┬───────────────┘
                                    │
        ┌──────────────┬────────────┼────────────┬──────────────┐
        │              │            │            │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌───▼──────────┐ ┌───▼──────┐
│  Stockfish   │ │   Maia     │ │   Lichess    │ │ Trap DB  │
│  (objective) │ │ (neural    │ │   Explorer   │ │ (known   │
│              │ │  human)    │ │ (empirical:  │ │ patterns)│
│              │ │            │ │ billions of  │ │          │
│              │ │            │ │ human games) │ │          │
└──────────────┘ └────────────┘ └──────────────┘ └──────────┘
       ▲              ▲ (fallback)     ▲                ▲
       └──────────────┼────────────────┼────────────────┘
                      │                │
                ┌─────▼────────────────▼──────────┐
                │   troll_engine.search           │
                │   • multipv candidates          │
                │   • re-rank by troll utility    │
                │   • + trap-db & empirical bonus │
                └─────────────────────────────────┘
```

### Three sources of human knowledge

The human-move predictor is layered, with cheap-and-empirical first:

1. **Lichess Opening Explorer** (`troll_engine/lichess_explorer.py`) —
   the free public API that aggregates every game on Lichess into a
   per-position move table. For in-book positions with ≥ 30 historical
   games at the target rating, we use the **empirical** move
   distribution directly — that's literally what humans of that
   strength played from this position. We also pull *outcome
   statistics* (win rate by side) and bonus the search when a move
   has a disproportionately good record at the target rating.

2. **Maia** (neural human-move model, `MaiaLc0Model`) — for positions
   the Explorer doesn't have. Maia's policy-only net predicts the move
   a human at rating R will play with ~50% top-1 accuracy. Requires
   Lc0 + Maia weights.

3. **Softmax-Stockfish** (`SoftmaxStockfishModel`) — final fallback if
   neither of the above is available. Captures "humans usually pick
   from Stockfish's top few" but misses pattern knowledge.

## Project layout

```
troll_engine/        # The brain — engine, human model, search, utility
  engine.py          # Stockfish UCI wrapper (multipv, eval, mate detection)
  human_model.py     # Maia interface + Lc0 implementation + fallback
  search.py          # Candidate generation & re-ranking pipeline
  utility.py         # Troll utility function (sacrifices, suffering, depth)
  analysis.py        # High-level API for the web layer
  trap_db.py         # Pattern matching against known trap positions

data/
  known_traps.json   # Hand-curated seed corpus (Légal's, Englund, etc.)
  mine_lichess.py    # Stream Lichess monthly dump, extract trap positions

app/
  server.py          # FastAPI endpoints (/analyse, /traps, /static)
  static/index.html  # Single-page UI with chessground board
  static/app.js      # Board, arrows, meters
  static/style.css   # Dark Tal-themed styling

colab_launcher.ipynb # One-click Colab: clones, installs, launches, tunnels
setup.sh             # Local install (Stockfish, Lc0, Maia weights)
```

## Running locally (with your own GPU)

Two commands. The first installs everything; the second launches the
server.

```bash
./setup.sh    # installs Stockfish, Lc0 (or hints at the CUDA build),
              # creates a venv, pip-installs, downloads Maia weights
./run.sh      # http://localhost:8000
```

`setup.sh` is idempotent — re-run any time to pick up new weights or
fix a partial install.

### GPU acceleration

Stockfish is CPU-only (fine — it's fast). The GPU only matters for
the **human-move predictor** (Maia, served by Lc0).

| OS | GPU | What to install |
|----|-----|-----------------|
| Linux | NVIDIA | Don't use `apt install lc0` (CPU-only). Grab the *linux-cuda* release from <https://github.com/LeelaChessZero/lc0/releases/latest>, extract, and symlink the binary into `/usr/local/bin/lc0`. `setup.sh` detects NVIDIA and prints the exact instructions. |
| macOS | Apple Silicon | `brew install lc0` — uses the Metal backend automatically. |
| Linux | no GPU | `apt install lc0` works. Maia inference will be CPU but still usable. |
| any | no Lc0 at all | The bot falls back to a softmax-Stockfish human-move predictor. The architecture stays correct; you just lose the *pattern* knowledge Maia learned from human games. |

Tune the engine via env vars in `run.sh`:

| Var | Default | What it does |
|---|---|---|
| `CD_THREADS` | 8 | Stockfish worker threads |
| `CD_DEPTH` | 18 | Search depth for candidate generation |
| `CD_SUB_DEPTH` | 14 | Depth for sub-position eval after a predicted reply |
| `CD_CANDIDATES` | 10 | How many candidate moves to re-rank |
| `CD_REPLIES` | 6 | How many opponent replies per candidate to sample |

On a beefy machine bump `CD_DEPTH` to 22 and `CD_CANDIDATES` to 14 — the
extra width finds deeper traps that shallow search misses.

## Running in Colab

Open `colab_launcher.ipynb` — it installs everything, downloads
weights, launches the server, and exposes a public URL via Cloudflare
tunnel.

## Next steps (the real fun)

- **Trap mining pipeline** — Lichess publishes ~100GB/month of games. We
  stream the PGN, find positions where a higher-rated player lost material
  in ≤6 plies after a sacrifice from a lower-rated player. Those are the
  *real* traps people fall for. The Opening Explorer is great for
  on-demand queries; offline mining lets us pre-compute the most-trap-y
  positions globally and ship them in the trap DB.
- **Personalized Maia** — fine-tune a Maia head on the specific opponent's
  Lichess/Chess.com history. The bot learns *your* opponent.
- **Time-trouble awareness** — humans crack faster in time pressure. Scale
  the troll bonus by remaining clock.
- **Masters DB**: the Explorer also has a separate `masters` database
  of OTB games. Useful as a sanity check — "what do GMs do here?" —
  but obviously a worse human-trap model than the rated-pool data.
