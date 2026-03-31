/*
 * engine_types.h -- Internal engine types (Turn, TurnHash, offset structs, etc.).
 */
#pragma once

#include "coord.h"

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════
using Turn = std::pair<Coord, Coord>;

struct TurnHash {
    size_t operator()(const Turn& t) const {
        auto h = std::hash<int64_t>{};
        return h(t.first) ^ (h(t.second) * 0x9e3779b97f4a7c15ULL);
    }
};

struct WinOff  { int d_idx, oq, or_; };
struct EvalOff { int d_idx, k, oq, or_; };
struct NbOff   { int dq, dr; };

struct SavedState {
    int8_t cur_player;
    int8_t moves_left;
    int8_t winner;
    bool   game_over;
};

struct UndoStep {
    Coord      cell;
    SavedState state;
    int8_t     player;
};

struct TTEntry {
    uint32_t key = 0;   // upper 32 bits of hash (verification)
    int16_t  depth = 0;
    int8_t   flag  = 0; // TT_EXACT / TT_LOWER / TT_UPPER
    double   score = 0;
    Turn     move  = {};
    bool     has_move = false;
};

struct TimeUp {};
