#!/usr/bin/env python3
"""Profile engine at varying NEIGHBOR_DIST to find where time goes.

Builds instrumented copies at NEIGHBOR_DIST = 2, 3, 4, runs the same
positions at a fixed depth, and reports per-component timing breakdowns.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time

ROOT = os.path.abspath(os.path.dirname(__file__))
CURRENT_DIR = os.path.join(ROOT, "current")
PROFILE_DIR = os.path.join(ROOT, "_profile_build")
VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python")

sys.path.insert(0, ROOT)
from game import HexGame, Player

# Hex neighbor offsets (distance <= 4, to generate positions for all variants)
_D2_OFFSETS = [
    (dq, dr)
    for dq in range(-2, 3)
    for dr in range(-2, 3)
    if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)
]


def generate_positions(num_positions, seed=42):
    rng = random.Random(seed)
    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < num_positions * 10:
        attempts += 1
        num_stones = rng.randint(8, 20)
        game = HexGame(win_length=6)
        for _ in range(num_stones):
            if game.game_over:
                break
            if not game.board:
                candidates = [(0, 0)]
            else:
                candidates = list({
                    (q + dq, r + dr)
                    for q, r in game.board
                    for dq, dr in _D2_OFFSETS
                    if (q + dq, r + dr) not in game.board
                })
            if not candidates:
                break
            game.make_move(*rng.choice(candidates))
        if not game.game_over:
            positions.append(game)
    return positions


def serialize_position(game):
    cells = []
    for (q, r), player in game.board.items():
        cells.append({"q": q, "r": r, "p": 1 if player == Player.A else 2})
    return {
        "cells": cells,
        "cur_player": 1 if game.current_player == Player.A else 2,
        "moves_left": game.moves_left_in_turn,
        "move_count": game.move_count,
    }


def build_variant(dist):
    """Build an instrumented engine at the given NEIGHBOR_DIST."""
    build_dir = os.path.join(PROFILE_DIR, f"dist{dist}")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Copy all source files (including subdirectories)
    shutil.copytree(CURRENT_DIR, build_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('build', '*.so', '*.pyd', '*.egg-info'))

    # Patch NEIGHBOR_DIST in engine/constants.h
    engine_path = os.path.join(build_dir, "engine", "constants.h")
    with open(engine_path) as fh:
        code = fh.read()
    code = code.replace(
        "static constexpr int    NEIGHBOR_DIST      = 2;",
        f"static constexpr int    NEIGHBOR_DIST      = {dist};",
    )
    # Add profiling counters (gated by PROFILE_ENGINE define)
    profile_header = r'''
// ── Profiling counters (injected by profile_neighbor.py) ──
#ifdef PROFILE_ENGINE
#include <atomic>
struct ProfileCounters {
    int64_t scoring_ns = 0;       // Time in candidate delta scoring
    int64_t make_undo_ns = 0;     // Time in _make_turn + _undo_turn
    int64_t threat_ns = 0;        // Time in threat detection / filtering
    int64_t total_search_ns = 0;  // Total _minimax + _quiescence time
    int64_t scoring_calls = 0;    // Number of times candidate scoring ran
    int64_t make_calls = 0;       // Number of _make_turn calls
    int64_t undo_calls = 0;       // Number of _undo_turn calls
    int64_t cand_count_sum = 0;   // Sum of candidate set sizes at scoring
    int64_t delta_calls = 0;      // Number of _move_delta calls
};
static ProfileCounters g_prof;

static inline int64_t _prof_now() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
#define PROF_START(var) int64_t var = _prof_now()
#define PROF_END(field, var) g_prof.field += (_prof_now() - (var))
#define PROF_INC(field) g_prof.field++
#define PROF_ADD(field, v) g_prof.field += (v)
#else
#define PROF_START(var)
#define PROF_END(field, var)
#define PROF_INC(field)
#define PROF_ADD(field, v)
#endif
'''
    # Insert after the #pragma once and includes but before the constants
    marker = "// ═══════════════════════════════════════════════════════════════════════\n//  Constants"
    code = code.replace(marker, profile_header + "\n" + marker)

    # Instrument _move_delta
    code = code.replace(
        "double _move_delta(int q, int r, bool is_a) const {",
        "double _move_delta(int q, int r, bool is_a) const {\n        PROF_INC(delta_calls);"
    )

    # Instrument candidate scoring in _minimax (the scoring block)
    code = code.replace(
        "                std::vector<std::pair<double, Coord>> scored;\n"
        "                scored.reserve(cands.size());\n"
        "                for (Coord c : cands) {\n"
        "                    scored.push_back({_move_delta(pack_q(c), pack_r(c), is_a) * dsign, c});\n"
        "                }\n"
        "                std::sort(scored.begin(), scored.end(),",
        "                PROF_INC(scoring_calls);\n"
        "                PROF_ADD(cand_count_sum, static_cast<int64_t>(cands.size()));\n"
        "                PROF_START(_pst);\n"
        "                std::vector<std::pair<double, Coord>> scored;\n"
        "                scored.reserve(cands.size());\n"
        "                for (Coord c : cands) {\n"
        "                    scored.push_back({_move_delta(pack_q(c), pack_r(c), is_a) * dsign, c});\n"
        "                }\n"
        "                std::sort(scored.begin(), scored.end(),"
    )
    # Close the scoring timer after the sort + cap + pair generation + threat filter
    code = code.replace(
        "                turns = _filter_turns_by_threats(turns);\n"
        "            }\n"
        "        }\n"
        "\n"
        "        if (turns.empty()) {",
        "                turns = _filter_turns_by_threats(turns);\n"
        "                PROF_END(scoring_ns, _pst);\n"
        "            }\n"
        "        }\n"
        "\n"
        "        if (turns.empty()) {"
    )

    # Instrument _make_turn — wrap the whole body with timing
    code = code.replace(
        "    int _make_turn(const Turn& turn, UndoStep steps[2]) {\n"
        "        int q1 = pack_q(turn.first),  r1 = pack_r(turn.first);\n"
        "        int q2 = pack_q(turn.second), r2 = pack_r(turn.second);",
        "    int _make_turn(const Turn& turn, UndoStep steps[2]) {\n"
        "        PROF_INC(make_calls);\n"
        "        PROF_START(_pmt);\n"
        "        int q1 = pack_q(turn.first),  r1 = pack_r(turn.first);\n"
        "        int q2 = pack_q(turn.second), r2 = pack_r(turn.second);"
    )
    # Close make_turn timer before each return
    code = code.replace(
        "        _make(q1, r1);\n"
        "        if (_game_over) return 1;",
        "        _make(q1, r1);\n"
        "        if (_game_over) { PROF_END(make_undo_ns, _pmt); return 1; }"
    )
    code = code.replace(
        "        _make(q2, r2);\n"
        "        return 2;\n"
        "    }",
        "        _make(q2, r2);\n"
        "        PROF_END(make_undo_ns, _pmt);\n"
        "        return 2;\n"
        "    }"
    )

    # Instrument _undo_turn
    code = code.replace(
        "    void _undo_turn(const UndoStep steps[], int n) {\n"
        "        for (int i = n - 1; i >= 0; i--)",
        "    void _undo_turn(const UndoStep steps[], int n) {\n"
        "        PROF_INC(undo_calls);\n"
        "        PROF_START(_put);\n"
        "        for (int i = n - 1; i >= 0; i--)"
    )
    code = code.replace(
        "                  steps[i].state, steps[i].player);\n"
        "    }",
        "                  steps[i].state, steps[i].player);\n"
        "        PROF_END(make_undo_ns, _put);\n"
        "    }"
    )

    with open(engine_path, "w") as fh:
        fh.write(code)

    # Patch minimax_bot.cpp to expose profiling counters
    bot_path = os.path.join(build_dir, "minimax_bot.cpp")
    with open(bot_path) as fh:
        bot_code = fh.read()

    # Add profile counter exposure to the module
    bot_code = bot_code.replace(
        'PYBIND11_MODULE(minimax_cpp, m) {',
        'PYBIND11_MODULE(minimax_cpp, m) {\n'
        '#ifdef PROFILE_ENGINE\n'
        '    m.def("get_profile", []() {\n'
        '        py::dict d;\n'
        '        d["scoring_ns"] = g_prof.scoring_ns;\n'
        '        d["make_undo_ns"] = g_prof.make_undo_ns;\n'
        '        d["threat_ns"] = g_prof.threat_ns;\n'
        '        d["total_search_ns"] = g_prof.total_search_ns;\n'
        '        d["scoring_calls"] = g_prof.scoring_calls;\n'
        '        d["make_calls"] = g_prof.make_calls;\n'
        '        d["undo_calls"] = g_prof.undo_calls;\n'
        '        d["cand_count_sum"] = g_prof.cand_count_sum;\n'
        '        d["delta_calls"] = g_prof.delta_calls;\n'
        '        return d;\n'
        '    });\n'
        '    m.def("reset_profile", []() {\n'
        '        g_prof = ProfileCounters{};\n'
        '    });\n'
        '#endif\n'
    )

    # Add PROFILE_ENGINE to compile flags
    setup_path = os.path.join(build_dir, "setup.py")
    with open(setup_path) as fh:
        setup_code = fh.read()
    setup_code = setup_code.replace(
        'extra_compile_args=["-O3", "-march=native", "-DNDEBUG"]',
        'extra_compile_args=["-O3", "-march=native", "-DNDEBUG", "-DPROFILE_ENGINE"]'
    )
    with open(setup_path, "w") as fh:
        fh.write(setup_code)
    with open(bot_path, "w") as fh:
        fh.write(bot_code)

    # Build
    print(f"  Building NEIGHBOR_DIST={dist}...", end=" ", flush=True)
    result = subprocess.run(
        [VENV_PYTHON, "setup.py", "build_ext", "--inplace"],
        cwd=build_dir, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        return False
    print("OK")
    return True


_WORKER = textwrap.dedent(r'''
    import sys, json, time

    bot_dir = sys.argv[1]
    max_depth = int(sys.argv[2])
    root_dir = sys.argv[3]

    sys.path.insert(0, root_dir)
    sys.path.insert(0, bot_dir)

    import minimax_cpp
    from game import HexGame, Player

    positions = json.loads(sys.stdin.read())
    bot = minimax_cpp.MinimaxBot(999.0)
    bot.max_depth = max_depth

    has_profile = hasattr(minimax_cpp, 'get_profile')
    if has_profile:
        minimax_cpp.reset_profile()

    results = []
    total_t0 = time.perf_counter()
    for pos in positions:
        game = HexGame(win_length=6)
        game.board = {}
        for c in pos["cells"]:
            game.board[(c["q"], c["r"])] = Player.A if c["p"] == 1 else Player.B
        game.current_player = Player.A if pos["cur_player"] == 1 else Player.B
        game.moves_left_in_turn = pos["moves_left"]
        game.move_count = pos["move_count"]

        bot._nodes = 0
        t0 = time.perf_counter()
        moves = bot.get_move(game)
        elapsed = time.perf_counter() - t0

        results.append({
            "moves": [[q, r] for q, r in moves],
            "time_ms": elapsed * 1000,
            "depth": bot.last_depth,
            "score": bot.last_score,
            "nodes": bot._nodes,
        })

    total_elapsed = time.perf_counter() - total_t0

    profile = {}
    if has_profile:
        profile = {k: int(v) for k, v in minimax_cpp.get_profile().items()}

    output = {
        "results": results,
        "total_ms": total_elapsed * 1000,
        "profile": profile,
    }
    print(json.dumps(output))
''')


def run_variant(dist, positions_json, max_depth):
    build_dir = os.path.join(PROFILE_DIR, f"dist{dist}")
    proc = subprocess.run(
        [VENV_PYTHON, "-c", _WORKER, build_dir, str(max_depth), ROOT],
        input=positions_json, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f"Error running dist={dist}:")
        print(proc.stderr)
        return None
    return json.loads(proc.stdout.strip())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile NEIGHBOR_DIST scaling")
    parser.add_argument("-n", "--num-positions", type=int, default=10)
    parser.add_argument("-d", "--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dists", type=str, default="2,3,4",
                        help="Comma-separated NEIGHBOR_DIST values")
    args = parser.parse_args()

    dists = [int(x) for x in args.dists.split(",")]

    # Generate positions
    print(f"Generating {args.num_positions} positions (seed={args.seed})...")
    positions = generate_positions(args.num_positions, seed=args.seed)
    serialized = json.dumps([serialize_position(g) for g in positions])
    print(f"  Got {len(positions)} positions\n")

    # Build variants
    os.makedirs(PROFILE_DIR, exist_ok=True)
    for dist in dists:
        if not build_variant(dist):
            sys.exit(1)
    print()

    # Run variants
    all_data = {}
    for dist in dists:
        print(f"Running NEIGHBOR_DIST={dist} at depth {args.depth}...")
        data = run_variant(dist, serialized, args.depth)
        if data is None:
            sys.exit(1)
        all_data[dist] = data
        total_ms = data["total_ms"]
        total_nodes = sum(r["nodes"] for r in data["results"])
        avg_depth = sum(r["depth"] for r in data["results"]) / len(data["results"])
        print(f"  {total_ms:.0f} ms total, {total_nodes:,} nodes, "
              f"avg depth {avg_depth:.1f}")

    # Report
    print(f"\n{'=' * 90}")
    print(f"  NEIGHBOR_DIST scaling profile (depth={args.depth}, "
          f"{len(positions)} positions)")
    print(f"{'=' * 90}\n")

    # Summary table
    hdr = f"{'dist':>4}  {'total_ms':>10}  {'nodes':>10}  {'nodes/s':>12}  " \
          f"{'avg_depth':>9}  {'avg_cands':>9}  {'speedup':>7}"
    print(hdr)
    print("-" * len(hdr))

    base_ms = all_data[dists[0]]["total_ms"]
    for dist in dists:
        d = all_data[dist]
        total_ms = d["total_ms"]
        total_nodes = sum(r["nodes"] for r in d["results"])
        avg_depth = sum(r["depth"] for r in d["results"]) / len(d["results"])
        nps = total_nodes / (total_ms / 1000) if total_ms > 0 else 0
        prof = d.get("profile", {})
        sc = prof.get("scoring_calls", 0)
        avg_cands = prof.get("cand_count_sum", 0) / sc if sc > 0 else 0
        speedup = base_ms / total_ms if total_ms > 0 else 0

        print(f"{dist:4d}  {total_ms:10.0f}  {total_nodes:10,}  "
              f"{nps:12,.0f}  {avg_depth:9.1f}  {avg_cands:9.1f}  "
              f"{speedup:7.2f}x")

    # Detailed profile breakdown
    print(f"\n{'─' * 90}")
    print("  Per-component breakdown (nanoseconds → milliseconds)\n")

    hdr2 = (f"{'dist':>4}  {'scoring_ms':>10}  {'mk/undo_ms':>10}  "
            f"{'other_ms':>10}  {'scoring%':>8}  {'mk/undo%':>8}  "
            f"{'other%':>8}  {'avg_cands':>9}  {'ns/delta':>9}")
    print(hdr2)
    print("-" * len(hdr2))

    for dist in dists:
        d = all_data[dist]
        prof = d.get("profile", {})
        total_ms = d["total_ms"]
        scoring_ns = prof.get("scoring_ns", 0)
        make_undo_ns = prof.get("make_undo_ns", 0)
        scoring_ms = scoring_ns / 1e6
        make_undo_ms = make_undo_ns / 1e6
        other_ms = total_ms - scoring_ms - make_undo_ms
        scoring_calls = prof.get("scoring_calls", 0)
        delta_calls = prof.get("delta_calls", 0)
        cand_sum = prof.get("cand_count_sum", 0)
        avg_cands = cand_sum / scoring_calls if scoring_calls > 0 else 0
        scoring_pct = (scoring_ms / total_ms * 100) if total_ms > 0 else 0
        mu_pct = (make_undo_ms / total_ms * 100) if total_ms > 0 else 0
        other_pct = (other_ms / total_ms * 100) if total_ms > 0 else 0
        ns_per_delta = scoring_ns / delta_calls if delta_calls > 0 else 0

        print(f"{dist:4d}  {scoring_ms:10.0f}  {make_undo_ms:10.0f}  "
              f"{other_ms:10.0f}  {scoring_pct:7.1f}%  {mu_pct:7.1f}%  "
              f"{other_pct:7.1f}%  {avg_cands:9.1f}  {ns_per_delta:9.1f}")

    # Per-position detail for each variant
    print(f"\n{'─' * 90}")
    print("  Per-position timing (ms)\n")

    # Header
    cols = "".join(f"  d={d:d} ms" for d in dists)
    print(f"{'pos':>3}  {'stones':>6}" + cols)
    print("-" * (12 + 10 * len(dists)))

    for i in range(len(positions)):
        stones = len(positions[i].board)
        vals = "".join(
            f"  {all_data[d]['results'][i]['time_ms']:7.1f}"
            for d in dists
        )
        print(f"{i:3d}  {stones:6d}{vals}")

    print(f"\n{'=' * 90}")

    # Cleanup note
    print(f"\nProfile builds in {PROFILE_DIR}/")
    print("Remove with: rm -rf _profile_build/")


if __name__ == "__main__":
    main()
