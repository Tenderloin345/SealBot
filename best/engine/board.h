/*
 * board.h -- MinimaxBot board operations: make/undo, move delta, eval tables.
 */
#pragma once

#include "bot.h"

namespace opt {

// ────────────────────────────────────────────────────────────────
//  Pattern table construction
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_build_eval_tables() {
    _eval_offsets.clear();
    for (int d = 0; d < 3; d++)
        for (int k = 0; k < _eval_length; k++)
            _eval_offsets.push_back({d, k, k * DIR_Q[d], k * DIR_R[d]});
    _pow3.resize(_eval_length);
    _pow3[0] = 1;
    for (int i = 1; i < _eval_length; i++)
        _pow3[i] = _pow3[i - 1] * 3;
}

// ────────────────────────────────────────────────────────────────
//  Incremental make / undo
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_make(int q, int r) {
    int8_t player = _cur_player;

    // Zobrist
    _hash ^= get_zobrist(q, r, player);

    int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;
    int qi = q + OFF, ri = r + OFF;

    // ── 6-cell windows ──
    bool won = false;
    if (player == P_A) {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.first++;
            if (counts.first >= 4) _hot_a.insert(wo.d_idx, sqi, sri);
            if (counts.first == WIN_LENGTH && counts.second == 0) won = true;
        }
    } else {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.second++;
            if (counts.second >= 4) _hot_b.insert(wo.d_idx, sqi, sri);
            if (counts.second == WIN_LENGTH && counts.first == 0) won = true;
        }
    }

    // ── N-cell eval windows ──
    const double* pv = _pv.data();
    for (const auto& eo : _eval_offsets) {
        int sqi = qi - eo.oq, sri = ri - eo.or_;
        int& slot = _wp[eo.d_idx][sqi][sri];
        int old_pi = slot;
        int new_pi = old_pi + cell_val * _pow3[eo.k];
        _eval_score += pv[new_pi] - pv[old_pi];
        slot = new_pi;
    }

    // ── Candidates ──
    Coord cell = pack(q, r);
    _cand_set.erase(cell);
    _rc_stack.push_back(_cand_rc[qi][ri]);
    _cand_rc[qi][ri] = 0;

    for (const auto& nb : g_nb_offsets) {
        int nq = q + nb.dq, nr = r + nb.dr;
        int nqi = nq + OFF, nri = nr + OFF;
        _cand_rc[nqi][nri]++;
        if (_board[nqi][nri] == 0)
            _cand_set.insert(pack(nq, nr));
    }

    // Place stone
    _board[qi][ri] = player;
    _board_cells.push_back(cell);
    _move_count++;

    if (won) {
        _winner    = player;
        _game_over = true;
    } else {
        _moves_left--;
        if (_moves_left <= 0) {
            _cur_player = (player == P_A) ? P_B : P_A;
            _moves_left = 2;
        }
    }
}

inline void MinimaxBot::_undo(int q, int r, const SavedState& st, int8_t player) {
    int qi = q + OFF, ri = r + OFF;

    // Remove stone
    _board[qi][ri] = 0;
    _board_cells.pop_back();
    _move_count--;
    _cur_player = st.cur_player;
    _moves_left = st.moves_left;
    _winner     = st.winner;
    _game_over  = st.game_over;

    // Zobrist
    _hash ^= get_zobrist(q, r, player);

    int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

    // ── 6-cell windows ──
    if (player == P_A) {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.first--;
            if (counts.first < 4) _hot_a.erase(wo.d_idx, sqi, sri);
        }
    } else {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.second--;
            if (counts.second < 4) _hot_b.erase(wo.d_idx, sqi, sri);
        }
    }

    // ── N-cell eval windows ──
    const double* pv = _pv.data();
    for (const auto& eo : _eval_offsets) {
        int sqi = qi - eo.oq, sri = ri - eo.or_;
        int& slot = _wp[eo.d_idx][sqi][sri];
        int old_pi = slot;
        int new_pi = old_pi - cell_val * _pow3[eo.k];
        _eval_score += pv[new_pi] - pv[old_pi];
        slot = new_pi;
    }

    // ── Candidates ──
    for (const auto& nb : g_nb_offsets) {
        int nq = q + nb.dq, nr = r + nb.dr;
        int nqi = nq + OFF, nri = nr + OFF;
        _cand_rc[nqi][nri]--;
        if (_cand_rc[nqi][nri] == 0)
            _cand_set.erase(pack(nq, nr));
    }
    int saved_rc = _rc_stack.back();
    _rc_stack.pop_back();
    if (saved_rc > 0) {
        Coord cell = pack(q, r);
        _cand_rc[qi][ri] = saved_rc;
        _cand_set.insert(cell);
    }
}

// ────────────────────────────────────────────────────────────────
//  Turn make / undo
// ────────────────────────────────────────────────────────────────
inline int MinimaxBot::_make_turn(const Turn& turn, UndoStep steps[2]) {
    int q1 = pack_q(turn.first),  r1 = pack_r(turn.first);
    int q2 = pack_q(turn.second), r2 = pack_r(turn.second);

    steps[0] = {turn.first, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
    _make(q1, r1);
    if (_game_over) return 1;

    steps[1] = {turn.second, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
    _make(q2, r2);
    return 2;
}

inline void MinimaxBot::_undo_turn(const UndoStep steps[], int n) {
    for (int i = n - 1; i >= 0; i--)
        _undo(pack_q(steps[i].cell), pack_r(steps[i].cell),
              steps[i].state, steps[i].player);
}

// ────────────────────────────────────────────────────────────────
//  Move delta
// ────────────────────────────────────────────────────────────────
inline double MinimaxBot::_move_delta(int q, int r, bool is_a) const {
    int8_t cell_val = is_a ? _cell_a : _cell_b;
    const double* pv = _pv.data();
    int qi = q + OFF, ri = r + OFF;
    double delta = 0.0;
    for (const auto& eo : _eval_offsets) {
        int old_pi = _wp[eo.d_idx][qi - eo.oq][ri - eo.or_];
        int new_pi = old_pi + cell_val * _pow3[eo.k];
        delta += pv[new_pi] - pv[old_pi];
    }
    return delta;
}

} // namespace opt
