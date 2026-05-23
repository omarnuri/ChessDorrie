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
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼──────┐    ┌──────────▼──────────┐  ┌────────▼────────┐
│  Stockfish   │    │   Maia (human       │  │   Trap DB       │
│  (objective) │    │   move predictor)   │  │ (known patterns)│
└──────────────┘    └─────────────────────┘  └─────────────────┘
       ▲                       ▲                      ▲
       └───────────────────────┼──────────────────────┘
                               │
                  ┌────────────▼───────────────┐
                  │   troll_engine.search      │
                  │   • multipv candidates     │
                  │   • re-rank by utility     │
                  │   • merge trap-db priors   │
                  └────────────────────────────┘
```

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

## Running locally

```bash
./setup.sh                # Installs Stockfish, downloads Maia weights
python -m app.server      # http://localhost:8000
```

## Running in Colab

Open `colab_launcher.ipynb` — it installs everything, downloads weights,
launches the server, and exposes a public URL via Cloudflare tunnel.

## Next steps (the real fun)

- **Trap mining pipeline** — Lichess publishes ~100GB/month of games. We
  stream the PGN, find positions where a higher-rated player lost material
  in ≤6 plies after a sacrifice from a lower-rated player. Those are the
  *real* traps people fall for.
- **Personalized Maia** — fine-tune a Maia head on the specific opponent's
  Lichess/Chess.com history. The bot learns *your* opponent.
- **Time-trouble awareness** — humans crack faster in time pressure. Scale
  the troll bonus by remaining clock.
- **Opening-book traps** — book moves are pre-computed. We don't need to
  search — just play the line.
