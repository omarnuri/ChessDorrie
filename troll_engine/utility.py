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

    # Setup-mode signal: largest sacrifice opportunity available 2 plies ahead
    trap_potential_cp: float = 0.0

    # Win/Draw/Loss probabilities from the side-to-move's POV after this
    # move (engines that report wdl). High win + low draw = sharp position
    # ripe for human error; high draw with similar cp = quiet, harder
    # to troll.
    wdl: tuple[float, float, float] | None = None


# Opponent-style profiles. Used to bias the utility weights.
#   "greedy"   — grabs every pawn, follows tactics greedily. Vulnerable to bait.
#   "balanced" — average human, no strong bias.
#   "cautious" — defensive, refuses speculative captures. Vulnerable to slow
#                squeezes where every move is subtly bad.
OPPONENT_STYLES = ("greedy", "balanced", "cautious")

# Bot playstyle — orthogonal to opponent style.
#   "direct" — take the troll-best move now; if a sac is available, play it.
#   "setup"  — prefer moves that BUILD UP toward sacrifices that become
#              available 2 plies later. Slightly discounts the immediate-sac
#              bonus in favour of moves whose top-1 future continuation has
#              a high-quality sacrificial candidate.
PLAYSTYLES = ("direct", "setup", "setup_deep")


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


def troll_utility(
    f: TrollFeatures, *, elo: int = 1500, style: str = "balanced",
    playstyle: str = "direct",
) -> UtilityOutput:
    """The composite scoring function.

    Weights are tuned for "Tal-style menace" — large bonuses for
    successful sacrifices, mild bonuses for human-factor exploitation,
    big penalties for unsafe play.

    `style` biases the weights based on perceived opponent psychology:

    * **greedy** — boost the sacrifice and human-factor terms; greedy
      opponents grab every offered piece and walk into prepared mates.
    * **balanced** — neutral.
    * **cautious** — diminish the sacrifice bonus (they won't take
      the bait) and instead reward positions where ALL of the
      opponent's plausible replies are subtly bad: cautious players
      prefer "safe-looking" moves and will obligingly walk into a
      slow squeeze.
    """
    notes: list[str] = []
    style = style if style in OPPONENT_STYLES else "balanced"
    playstyle = playstyle if playstyle in PLAYSTYLES else "direct"

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
        # Style modulation
        if style == "greedy":
            sacrifice_bonus *= 1.6   # they'll take the bait
        elif style == "cautious":
            sacrifice_bonus *= 0.5   # they probably won't take
        # Playstyle modulation: setup-mode prefers DELAYED sacs over
        # immediate ones, so discount the current-move sac bonus.
        if playstyle in ("setup", "setup_deep"):
            sacrifice_bonus *= 0.7
    score += sacrifice_bonus

    # --- 3b. Setup-mode: trap-potential bonus ---------------------------
    # Reward moves that lead to positions where a sacrifice becomes
    # available within 2 plies. Always present as a signal; only
    # contributes meaningfully in setup playstyle.
    if f.trap_potential_cp > 0:
        if playstyle == "setup_deep":
            weight = 1.0
        elif playstyle == "setup":
            weight = 0.8
        else:
            weight = 0.15
        tp_bonus = min(400.0, f.trap_potential_cp * weight)
        score += tp_bonus
        if playstyle in ("setup", "setup_deep") and tp_bonus > 50:
            horizon = "4 plies" if playstyle == "setup_deep" else "2 plies"
            notes.append(
                f"🪤 sets up a sacrifice {horizon} ahead (+{tp_bonus:.0f}cp)"
            )

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

    # --- 6. Mate-with-style --------------------------------------------
    # A quick mate AFTER a sacrifice is the Tal apex. A clean tactical
    # mate without sacrificial spice is fine but less iconic.
    if f.is_mate_for_us and f.mate_in is not None:
        if f.is_sacrifice:
            # Sacrifice → mate: jackpot. Bonus inversely proportional to mate length.
            bonus = 1200.0 / max(1, f.mate_in)
            score += bonus
            notes.append(f"🔥 Mate-in-{f.mate_in} after a sacrifice — Tal-class brilliancy")
        elif f.mate_in <= 2:
            # Mate-in-1 or mate-in-2 with no sacrifice — still good, mild boost
            score += 200
            notes.append(f"Forced mate in {f.mate_in}")
        elif 3 <= f.mate_in <= 5 and f.expected_material_cp < 200:
            # Quick clean mate, no material extracted — tiny penalty
            score -= 40
            notes.append("Quick clean mate — could be flashier")

    # --- 7. Forcing-reply bonus ----------------------------------------
    # If the reply distribution is peaked on one move (low entropy), and
    # the opponent's *other* replies hemorrhage material, that's a trap
    # they have to find their way out of.
    if f.reply_entropy < 0.6 and f.expected_eval_loss_to_opponent > 100:
        score += 80
        notes.append("Single-reply trap — opponent must find THE move")

    # --- 7b. Slow-squeeze bonus (cautious opponents) -------------------
    # When the opponent has many similar-looking replies that are ALL
    # subtly bad, a cautious player walks into one without complaint.
    # Reward positions where (a) reply entropy is high, (b) every reply
    # loses some material in expectation. This is the "no safe square"
    # trap that grinds tinfoil-hat defenders into dust.
    if style == "cautious":
        if f.reply_entropy > 1.0 and f.expected_eval_loss_to_opponent > 40:
            score += 120
            notes.append("🕸 Slow squeeze — every retreat is subtly bad")
        # Cautious players over-defend, so positional pressure on objective
        # eval translates better than tactical sharpness.
        score += max(0, f.expected_eval - 100) * 0.2

    # Style modulation on human-factor exploitation.
    if style == "greedy":
        # Greedy players also miss in non-sacrificial positions —
        # double-count the human-factor bonus a bit.
        score += max(0, human_factor) * 0.4

    # --- 7c. WDL sharpness bonus ----------------------------------------
    # Engines that report win/draw/loss probabilities give us a much
    # richer signal than centipawn eval. A position with low draw
    # probability is *sharp* — small mistakes cascade into wins or
    # losses. Humans crack faster in sharp positions, so we bonus.
    #
    # A position with high draw probability is *technical* — even with
    # a small cp advantage it's hard to convert against a defending
    # human. Discount slightly.
    if f.wdl is not None:
        w, d, l = f.wdl
        # Sharpness: 1 - draw_prob, scaled. 0.95 draw → sharpness 0.05.
        # 0.10 draw → sharpness 0.90.
        sharpness = max(0.0, 1.0 - d)
        # Asymmetry penalty: if loss > win, it's sharp against us — bad.
        if w > l:
            sharp_bonus = 60.0 * sharpness * (w - l)
            if sharp_bonus > 10:
                score += sharp_bonus
                if sharpness > 0.6 and (w - l) > 0.2:
                    notes.append(
                        f"💥 sharp position ({w*100:.0f}/{d*100:.0f}/{l*100:.0f} wdl) — humans crack"
                    )
        elif d > 0.7 and abs(f.expected_eval) < 100:
            # Drawish balanced — discount; troll utility hates quiet
            score -= 30
            notes.append("🥱 drawish — hard to make humans crack")

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
