"""Troll utility function.

Given the per-move features produced by `search.evaluate_candidates`,
compute the composite `troll_score` that the search ranks by, plus the
derived UI metrics (anger probability, human factor, etc.).

The whole soul of the bot lives in these weights. Tune freely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Anything worse than this in the worst-case reply line → drop the move.
# We're trolling, not throwing.
SAFETY_FLOOR_CP = -150

# Mate represented as values near ±100_000 in engine.py.
_MATE_BAND = 50_000


@dataclass
class TrollFeatures:
    """Numbers handed to the utility function per candidate move."""

    # Stockfish view of the move itself
    objective_eval: float        # cp from bot's POV after our move, opp best play
    objective_rank: int          # 1 = engine's #1 move
    objective_best_eval: float   # cp of engine's #1 candidate (to compute drop)

    # Human-aware view (averaged over opponent's predicted replies)
    expected_eval: float
    worst_case_eval: float
    best_case_eval: float
    expected_material_cp: float  # net material delta vs. before our move
    reply_entropy: float         # natural-log entropy of reply distribution

    # Sacrifice markers
    is_sacrifice: bool
    sacrifice_value_cp: int      # how much material we're risking

    # Anger components (computed during search)
    expected_eval_loss_to_opponent: float  # cp the opponent gives us by mis-playing
    prob_opponent_blunders: float          # P(opp plays a move losing ≥150cp)

    # Mate info
    is_mate_for_us: bool
    mate_in: int | None


@dataclass
class UtilityOutput:
    troll_score: float
    human_factor: float          # |expected - objective|, cp
    anger_probability: float     # P(opp plays something materially bad)
    sacrifice_bonus: float       # for UI display
    notes: list[str]


def _is_mate_score(cp: float) -> bool:
    return abs(cp) > _MATE_BAND


def shannon_entropy(probs: list[float]) -> float:
    """Natural-log entropy of a probability distribution.
    Used as a measure of how "forcing" the resulting position is."""
    total = sum(probs) or 1.0
    h = 0.0
    for p in probs:
        q = p / total
        if q > 0:
            h -= q * math.log(q)
    return h


def troll_utility(f: TrollFeatures, *, elo: int = 1500) -> UtilityOutput:
    """The composite scoring function.

    Weights are tuned for "Tal-style menace" — large bonuses for
    successful sacrifices, mild bonuses for human-factor exploitation,
    big penalties for unsafe play, mild penalty for quick clean mates
    that skip the material humiliation.
    """
    notes: list[str] = []

    # --- 1. Safety floor ------------------------------------------------
    # Reject moves that lose badly against any plausible reply (unless
    # we're delivering mate). In an already-losing position we relax the
    # threshold so the bot can still swindle from −5; in a winning
    # position we hold the line at −1.5.
    safety_threshold = min(SAFETY_FLOOR_CP, f.objective_best_eval - 200)
    if f.worst_case_eval < safety_threshold and not f.is_mate_for_us:
        return UtilityOutput(
            troll_score=-1e9,
            human_factor=0.0,
            anger_probability=0.0,
            sacrifice_bonus=0.0,
            notes=["unsafe — at least one plausible reply leaves us losing"],
        )

    # --- 2. Base score: expected eval after a human reply ---------------
    score = f.expected_eval

    # --- 3. Sacrifice bonus --------------------------------------------
    # The bigger the sac AND the smaller the expected material loss, the
    # more spectacular. If we sac a piece but our expected position is
    # *still* materially up, that's the Tal jackpot.
    sacrifice_bonus = 0.0
    if f.is_sacrifice:
        sac_cp = f.sacrifice_value_cp
        if f.expected_material_cp >= 0:
            # Material-neutral or positive after their typical reply → jackpot
            sacrifice_bonus = 3.5 * sac_cp
            notes.append(f"⚡ Tal-grade sacrifice: humans usually miss the refutation ({sac_cp}cp risked)")
        elif f.expected_eval > 200:
            # Down material but clearly winning position → positional sac
            sacrifice_bonus = 1.5 * sac_cp
            notes.append(f"Speculative sac with strong positional compensation ({sac_cp}cp)")
        elif f.expected_eval > 0:
            sacrifice_bonus = 0.3 * sac_cp
            notes.append("Sac with marginal compensation")
    score += sacrifice_bonus

    # --- 4. Human-factor amplification ---------------------------------
    # If exploiting human-ness gains us material vs. objective best play,
    # amplify it. We *want* moves that surprise humans more than engines.
    human_factor = f.expected_eval - f.objective_eval
    if human_factor > 0:
        # Reward divergence, but with diminishing returns
        score += min(300.0, human_factor * 0.7)
        if human_factor > 100:
            notes.append(f"Exploits human factor: +{human_factor:.0f}cp vs. perfect play")

    # --- 5. Objective-quality floor (don't pick obviously bad moves) ---
    # If SF says this loses ≥1 full pawn vs. its top pick, and we're not
    # getting compensation through the human factor or a sac, penalize.
    drop = f.objective_best_eval - f.objective_eval
    if drop > 100 and not f.is_sacrifice and human_factor < 50:
        score -= 0.8 * drop
        notes.append(f"⚠ engine disapproves (−{drop:.0f}cp) and the trap isn't strong")

    # --- 6. Anti-efficient-mate -----------------------------------------
    # We're in this for the humiliation. A clean mate-in-3 with no
    # material gain is boring next to "win their queen, then mate in 12".
    if f.is_mate_for_us and f.mate_in is not None and 1 <= f.mate_in <= 3:
        if f.expected_material_cp < 300:
            score -= 250
            notes.append("Short mate — prefer to win material first if possible")

    # --- 7. Forcing-reply bonus ----------------------------------------
    # If the reply distribution is peaked on one move (low entropy), and
    # the opponent's *other* replies hemorrhage material, that's a trap
    # they have to find their way out of.
    if f.reply_entropy < 0.6 and f.expected_eval_loss_to_opponent > 100:
        score += 80
        notes.append("Single-reply trap — opponent must find THE move")

    # --- 8. Boring-move dampener ---------------------------------------
    # If a move is just SF's best with no sacrificial spice and no human
    # exploitation, slightly dampen it relative to alternatives. We want
    # the bot to choose the spicier of two equal options.
    if not f.is_sacrifice and human_factor < 20 and f.objective_rank == 1:
        score -= 10  # tiny — only breaks ties

    # --- Derived UI metrics --------------------------------------------
    anger_probability = max(0.0, min(1.0, f.prob_opponent_blunders))

    return UtilityOutput(
        troll_score=score,
        human_factor=human_factor,
        anger_probability=anger_probability,
        sacrifice_bonus=sacrifice_bonus,
        notes=notes,
    )
