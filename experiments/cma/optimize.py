"""CMA-ES optimization of SealBot pattern evaluation weights.

Optimizes the 364 free parameters (from 729 total, halved by player-swap
symmetry) using CMA-ES. Fitness is measured by win rate against the
baseline (hardcoded best/ weights) over a batch of games.

Usage:
    python optimize.py                     # defaults
    python optimize.py --games 30 --popsize 80
    python optimize.py --resume            # resume from checkpoint
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import cma
import numpy as np
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pkl")
POOL_DIR = os.path.join(OUTPUT_DIR, "pool")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from symmetry import (free_to_full, full_to_free, load_baseline, load_current,
                       save_pattern_data_h, CURRENT_PATTERN_DATA_PATH)


# ── Statistics (mirrors evaluate.py) ────────────────────────────────────────

def _norm_sf(x):
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    return poly * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _elo(score):
    if score <= 0.0: return float('-inf')
    if score >= 1.0: return float('inf')
    return -400 * math.log10(1.0 / score - 1.0)


def win_rate_stats(wins, losses, draws):
    """Wilson CI, p-value, and Elo from W/L/D counts."""
    n = wins + losses + draws
    if n == 0:
        return {"wr": 0.5, "ci_lo": 0.0, "ci_hi": 1.0, "p": 1.0, "elo": 0, "n": 0}
    score = wins + 0.5 * draws
    p_hat = score / n
    z = 1.96
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    ci_lo = max(0.0, centre - spread)
    ci_hi = min(1.0, centre + spread)
    z_obs = (score - 0.5 * n) / math.sqrt(0.25 * n) if n > 0 else 0
    p_value = 2 * _norm_sf(abs(z_obs)) if n > 0 else 1.0
    return {"wr": p_hat, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p": p_value, "elo": _elo(p_hat), "n": n}


# ── Global config (set by main, read by workers) ───────────────────────────

_CFG = {}


def _init_worker(cfg):
    """Initializer for each pool worker -- import the C++ module once."""
    global _CFG
    _CFG = cfg
    # Import here so each worker process has its own module instance
    sys.path.insert(0, cfg["script_dir"])
    sys.path.insert(0, cfg["root_dir"])



def _play_game(bot_a, bot_b, time_limit, max_moves=200):
    """Play one game between two bot engines. Returns the winner."""
    from game import HexGame, Player

    game = HexGame(win_length=6)
    bots = {Player.A: bot_a, Player.B: bot_b}
    total = 0

    while not game.game_over and total < max_moves:
        player = game.current_player
        bot = bots[player]
        bot.time_limit = time_limit
        moves = bot.get_move(game)

        if not moves:
            return Player.B if player == Player.A else Player.A

        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                return Player.B if player == Player.A else Player.A
        total += len(moves)

    return game.winner


def _play_single_game(args):
    """Play one game. Returns (candidate_idx, w, l, d) as ints.

    cfg must contain either 'opponent_weights' (fixed opponent) or
    'opponent_pool' (random choice per game).
    """
    free_params, candidate_idx, game_idx, cfg = args

    sys.path.insert(0, cfg["script_dir"])
    sys.path.insert(0, cfg["root_dir"])
    from game import Player
    import cma_minimax_cpp

    swapped = game_idx % 2 == 1
    tl = random.uniform(cfg["time_limit"], cfg.get("time_limit_max", cfg["time_limit"]))

    full_params = free_to_full(free_params, single_color=cfg.get("single_color", False))
    candidate = cma_minimax_cpp.MinimaxBot(tl)
    candidate.load_patterns(full_params)
    baseline = cma_minimax_cpp.MinimaxBot(tl)

    if "opponent_weights" in cfg:
        baseline.load_patterns(cfg["opponent_weights"])
    elif cfg.get("opponent_pool"):
        baseline.load_patterns(random.choice(cfg["opponent_pool"]))

    if swapped:
        bot_a, bot_b = baseline, candidate
    else:
        bot_a, bot_b = candidate, baseline

    try:
        winner = _play_game(bot_a, bot_b, tl)
    except Exception:
        winner = Player.NONE

    candidate_is = Player.A if not swapped else Player.B
    if winner == candidate_is:
        return (candidate_idx, 1, 0, 0)
    elif winner == Player.NONE:
        return (candidate_idx, 0, 0, 1)
    else:
        return (candidate_idx, 0, 1, 0)


def _run_promotion_eval(best_free, opponent_weights, cfg, num_games, num_workers,
                         desc="Promotion"):
    """Evaluate best vs opponent, one game per task for even load balancing."""
    game_cfg = dict(cfg, opponent_weights=opponent_weights)
    tasks = [(best_free, 0, i, game_cfg) for i in range(num_games)]

    with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        results = list(tqdm(pool.imap(_play_single_game, tasks),
                            total=num_games, desc=desc, leave=False))

    total_w = sum(r[1] for r in results)
    total_l = sum(r[2] for r in results)
    total_d = sum(r[3] for r in results)
    return total_w, total_l, total_d


# ── Main optimization loop ─────────────────────────────────────────────────

def _load_pool(pool_dir, baseline_full):
    """Load opponent weight files from pool dir. Returns list of full-729 lists."""
    pool = [baseline_full]  # always include the original baseline
    if os.path.isdir(pool_dir):
        for name in sorted(os.listdir(pool_dir)):
            if not name.endswith(".h"):
                continue
            path = os.path.join(pool_dir, name)
            try:
                import re
                with open(path) as f:
                    text = f.read()
                match = re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}", text)
                if match:
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                      match.group(1))
                    vals = [float(x) for x in nums]
                    if len(vals) == 729:
                        pool.append(vals)
            except Exception:
                pass
    return pool


def _save_to_pool(pool_dir, full_params, gen):
    """Save weights to pool directory."""
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, f"gen_{gen:04d}.h")
    save_pattern_data_h(full_params, path)
    return path


def run(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load starting point
    current_full = load_current()
    x0 = full_to_free(current_full, single_color=args.single_color)
    print(f"Free parameters: {len(x0)} (from {len(current_full)} total)"
          f"{' [single-color only]' if args.single_color else ''}")
    print(f"Parameter stats: median |x|={np.median(np.abs(x0)):.0f}, "
          f"mean |x|={np.mean(np.abs(x0)):.0f}, max |x|={np.max(np.abs(x0)):.0f}")

    # Load opponent pool
    opponent_pool = _load_pool(POOL_DIR, current_full)
    print(f"Opponent pool: {len(opponent_pool)} bots"
          f" (1 current + {len(opponent_pool)-1} from {POOL_DIR})")

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["popsize"] = args.popsize
    opts["maxiter"] = args.max_gen
    opts["seed"] = args.seed
    opts["verb_disp"] = 1
    opts["verb_filenameprefix"] = os.path.join(OUTPUT_DIR, "outcma_")
    opts["verb_log"] = 1
    opts["tolfun"] = 1e-6

    # Resume or start fresh
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        es = state["es"]
        best_fitness = state["best_fitness"]
        best_free = state["best_free"]
        gen_offset = state["generation"]
        promotion_count = state.get("promotion_count", 0)
        print(f"  Resuming at generation {gen_offset}, best fitness {best_fitness:.4f}"
              f", promotions: {promotion_count}")
    else:
        # Clean start: wipe all previous output
        import shutil
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        opponent_pool = [current_full]  # reset pool to just current
        print(f"  Clean start (wiped {OUTPUT_DIR})")

        es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)
        best_fitness = 0.0   # worst possible (win rate = 0)
        best_free = x0.copy()
        gen_offset = 0
        promotion_count = 0

    # Worker config
    time_limit_max = args.time_limit_max or args.time_limit
    cfg = {
        "num_games": args.games,
        "time_limit": args.time_limit,
        "time_limit_max": time_limit_max,
        "script_dir": SCRIPT_DIR,
        "root_dir": ROOT_DIR,
        "single_color": args.single_color,
        "opponent_pool": opponent_pool,
    }

    num_workers = args.workers or mp.cpu_count()
    tl_str = (f"{args.time_limit}s" if time_limit_max == args.time_limit
              else f"{args.time_limit}–{time_limit_max}s")
    print(f"\nCMA-ES: sigma0={args.sigma0}, popsize={args.popsize}, "
          f"games/eval={args.games}, time_limit={tl_str}")
    print(f"Workers: {num_workers}, max generations: {args.max_gen}")
    print(f"Promote at {args.promote_threshold:.0%} WR ({args.promote_games} games vs current/)")
    print(f"Output: {OUTPUT_DIR}\n")

    # CSV log
    csv_path = os.path.join(OUTPUT_DIR, "log.csv")
    csv_fields = ["gen", "best_wr", "gen_best_wr", "mean_wr",
                  "sigma", "gen_time", "total_time"]
    write_header = not (args.resume and os.path.exists(csv_path))
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        csv_writer.writeheader()

    gen = gen_offset
    t_start = time.time()

    try:
        while not es.stop():
            gen += 1
            t_gen = time.time()

            # Ask for candidate solutions
            solutions = es.ask()

            # Evaluate in parallel -- one game per task for smooth progress
            num_games = cfg["num_games"]
            tasks = [(sol, ci, gi, cfg)
                     for ci, sol in enumerate(solutions)
                     for gi in range(num_games)]

            with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                game_results = list(tqdm(pool.imap(_play_single_game, tasks),
                                         total=len(tasks), desc=f"Gen {gen}",
                                         leave=False))

            # Aggregate per candidate
            n_candidates = len(solutions)
            wins = [0] * n_candidates
            losses = [0] * n_candidates
            draws = [0] * n_candidates
            for ci, w, l, d in game_results:
                wins[ci] += w
                losses[ci] += l
                draws[ci] += d

            results = []
            for ci in range(n_candidates):
                wr = (wins[ci] + 0.5 * draws[ci]) / num_games
                results.append((-wr, wins[ci], losses[ci], draws[ci]))

            fitnesses = [r[0] for r in results]
            es.tell(solutions, fitnesses)

            # Track best
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            gen_best_free = np.array(solutions[gen_best_idx])
            best_changed = False
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_free = gen_best_free
                best_changed = True

            # Save every generation's best for later evaluation
            gen_full = free_to_full(gen_best_free,
                                    single_color=args.single_color)
            _save_to_pool(POOL_DIR, gen_full, gen)

            elapsed_gen = time.time() - t_gen
            elapsed_total = time.time() - t_start

            # Per-generation summary
            gen_w = sum(r[1] for r in results)
            gen_l = sum(r[2] for r in results)
            gen_d = sum(r[3] for r in results)
            gen_stats = win_rate_stats(gen_w, gen_l, gen_d)

            print(f"  Gen {gen}: best_wr={-best_fitness:.1%}, "
                  f"gen_best_wr={-gen_best_fit:.1%}, "
                  f"pop: {gen_w}W/{gen_l}L/{gen_d}D "
                  f"(elo {gen_stats['elo']:+.0f}), "
                  f"sigma={es.sigma:.1f}, "
                  f"{elapsed_gen:.0f}s"
                  f"{'  *new best*' if best_changed else ''}"
                  f"  [{len(opponent_pool)} opps]")

            # CSV row (validation fields filled below if applicable)
            row = {
                "gen": gen, "best_wr": f"{-best_fitness:.4f}",
                "gen_best_wr": f"{-gen_best_fit:.4f}",
                "mean_wr": f"{-np.mean(fitnesses):.4f}",
                "sigma": f"{es.sigma:.2f}",
                "gen_time": f"{elapsed_gen:.0f}",
                "total_time": f"{elapsed_total:.0f}",
            }

            csv_writer.writerow(row)
            csv_file.flush()

            # Checkpoint every generation
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({
                    "es": es,
                    "best_fitness": best_fitness,
                    "best_free": best_free,
                    "generation": gen,
                    "promotion_count": promotion_count,
                }, f)

            # Save best pattern_data.h every generation
            best_full = free_to_full(best_free, single_color=args.single_color)
            out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
            save_pattern_data_h(best_full, out_path)

            # ── Promotion check: evaluate best vs current/ ──
            if best_changed and args.promote_games > 0:
                print(f"  Promotion eval ({args.promote_games} games vs current/)...")
                t_promo = time.time()
                pw, pl, pd = _run_promotion_eval(
                    best_free, current_full, cfg,
                    args.promote_games, num_workers)
                ps = win_rate_stats(pw, pl, pd)
                print(f"    {pw}W/{pl}L/{pd}D = {ps['wr']:.1%} "
                      f"[{ps['ci_lo']:.1%}, {ps['ci_hi']:.1%}] "
                      f"Elo: {ps['elo']:+.0f} p={ps['p']:.4f} "
                      f"({time.time()-t_promo:.0f}s)")

                if ps['wr'] >= args.promote_threshold:
                    promotion_count += 1
                    best_full = free_to_full(best_free,
                                             single_color=args.single_color)
                    save_pattern_data_h(best_full, CURRENT_PATTERN_DATA_PATH)
                    current_full = best_full

                    # Check total progress vs best/
                    best_baseline = load_baseline()
                    bw, bl, bd = _run_promotion_eval(
                        best_free, best_baseline, cfg, 100, num_workers,
                        desc="vs best/")
                    bs = win_rate_stats(bw, bl, bd)
                    print(f"  vs best/: {bw}W/{bl}L/{bd}D = {bs['wr']:.1%} "
                          f"[{bs['ci_lo']:.1%}, {bs['ci_hi']:.1%}] "
                          f"Elo: {bs['elo']:+.0f} p={bs['p']:.4f}")

                    print(f"\n  {'='*50}")
                    print(f"  PROMOTED to current/ (#{promotion_count})")
                    print(f"  Restarting CMA-ES with new baseline")
                    print(f"  {'='*50}\n")

                    # Restart CMA-ES
                    x0 = best_free.copy()
                    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)
                    best_fitness = 0.0
                    best_free = x0.copy()

                    # Reset pool and config to new current
                    opponent_pool = [current_full]
                    cfg["opponent_pool"] = opponent_pool

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        csv_file.close()

    # Final save
    best_full = free_to_full(best_free)
    out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
    save_pattern_data_h(best_full, out_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  CMA-ES finished after {gen - gen_offset} generations ({elapsed/60:.0f} min)")
    print(f"  Promotions: {promotion_count}")
    print(f"  Best win rate vs current: {-best_fitness:.1%}")
    print(f"  CSV log:  {csv_path}")
    print(f"  Weights:  {out_path}")
    print(f"\n  To use these weights:")
    print(f"    cp {out_path} ../../current/pattern_data.h")
    print(f"    cd ../.. && make rebuild")
    print(f"    python evaluate.py -n 100 -t 0.1")
    print(f"{'='*60}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization of SealBot pattern weights")

    parser.add_argument("--games", type=int, default=20,
                        help="Games per fitness evaluation (default: 20)")
    parser.add_argument("--time-limit", type=float, default=0.02,
                        help="Min seconds per move during evaluation (default: 0.02)")
    parser.add_argument("--time-limit-max", type=float, default=None,
                        help="Max seconds per move (default: same as --time-limit)")
    parser.add_argument("--popsize", type=int, default=50,
                        help="CMA-ES population size (default: 50)")
    parser.add_argument("--sigma0", type=float, default=50.0,
                        help="CMA-ES initial step size (default: 50.0)")
    parser.add_argument("--max-gen", type=int, default=500,
                        help="Maximum generations (default: 500)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: cpu_count)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--single-color", action="store_true",
                        help="Only optimize single-color patterns (63 params instead of 364)")
    parser.add_argument("--promote-threshold", type=float, default=0.65,
                        help="Win rate to promote best to current/ and restart (default: 0.65)")
    parser.add_argument("--promote-games", type=int, default=400,
                        help="Games for promotion evaluation (default: 400)")

    run(parser.parse_args())
