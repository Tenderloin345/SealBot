"""Ternary mirror symmetry utilities for pattern weights.

The evaluation uses 729 weights (3^6 patterns over 6-cell windows).
Each cell is encoded as: 0=empty, 1=player_A, 2=player_B.

Player-swap symmetry means pv[i] = -pv[mirror(i)], where mirror(i)
swaps digits 1<->2 in the base-3 representation. This halves the
search space to 364 free parameters (plus pv[0]=0 fixed).
"""

import os
import re

import numpy as np

PATTERN_LENGTH = 6
PATTERN_COUNT = 3 ** PATTERN_LENGTH  # 729

BEST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "best")
PATTERN_DATA_PATH = os.path.join(BEST_DIR, "pattern_data.h")
CURRENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "current")
CURRENT_PATTERN_DATA_PATH = os.path.join(CURRENT_DIR, "pattern_data.h")


def mirror(i: int, length: int = PATTERN_LENGTH) -> int:
    """Swap digits 1<->2 in the base-3 representation of index i."""
    result = 0
    power = 1
    for _ in range(length):
        digit = i % 3
        if digit == 1:
            digit = 2
        elif digit == 2:
            digit = 1
        result += digit * power
        power *= 3
        i //= 3
    return result


def free_indices(n: int = PATTERN_COUNT) -> list[int]:
    """Return indices where i < mirror(i). These are the 364 free parameters."""
    indices = []
    for i in range(n):
        m = mirror(i)
        if i < m:
            indices.append(i)
    return indices


# Precompute once at import time
_FREE_INDICES = free_indices()
_MIRROR_MAP = {i: mirror(i) for i in _FREE_INDICES}


def reverse_pattern(i: int, length: int = PATTERN_LENGTH) -> int:
    """Reverse the base-3 digits of index i (spatial reversal of the window)."""
    digits = []
    x = i
    for _ in range(length):
        digits.append(x % 3)
        x //= 3
    result = 0
    for d in digits:
        result = result * 3 + d
    return result


def single_color_free_indices(n: int = PATTERN_COUNT) -> list[int]:
    """Return single-color indices with reversal symmetry applied.

    Only patterns with 0s and 1s (no opponent pieces), excluding all-empty.
    Reversal symmetry: pv[i] = pv[reverse(i)] since a line reads the same
    both ways. Returns 35 indices for length 6.
    """
    indices = []
    for i in range(1, n):
        x = i
        ok = True
        for _ in range(PATTERN_LENGTH):
            if x % 3 == 2:
                ok = False
                break
            x //= 3
        if ok and i <= reverse_pattern(i):
            indices.append(i)
    return indices


_SC_FREE_INDICES = single_color_free_indices()
_SC_MIRROR_MAP = {i: mirror(i) for i in _SC_FREE_INDICES}
_SC_REVERSE_MAP = {i: reverse_pattern(i) for i in _SC_FREE_INDICES}


def free_to_full(free_params: np.ndarray, single_color: bool = False) -> list[float]:
    """Expand free params to full 729 vector via symmetry.

    pv[0] = 0 (all-empty pattern, neutral by definition).
    For each free index i with mirror j: pv[i] = param, pv[j] = -param.

    If single_color=True, applies both player-swap and reversal symmetry.
    Each free param sets 4 entries: i, reverse(i), mirror(i), mirror(reverse(i)).
    Mixed patterns are zeroed out. 35 free params for length 6.
    """
    full = [0.0] * PATTERN_COUNT
    if single_color:
        for k, i in enumerate(_SC_FREE_INDICES):
            v = float(free_params[k])
            rev = _SC_REVERSE_MAP[i]
            full[i] = v
            full[rev] = v
            full[mirror(i)] = -v
            full[mirror(rev)] = -v
    else:
        for k, i in enumerate(_FREE_INDICES):
            j = _MIRROR_MAP[i]
            full[i] = float(free_params[k])
            full[j] = -float(free_params[k])
    return full


def full_to_free(full_params, single_color: bool = False) -> np.ndarray:
    """Extract free params from a full 729 vector."""
    indices = _SC_FREE_INDICES if single_color else _FREE_INDICES
    return np.array([full_params[i] for i in indices])


def load_weights(path: str) -> list[float]:
    """Parse a pattern_data.h file and return the 729 values."""
    with open(path) as f:
        text = f.read()
    match = re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}", text)
    if not match:
        raise RuntimeError(f"Could not parse {path}")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
    values = [float(x) for x in nums]
    if len(values) != PATTERN_COUNT:
        raise RuntimeError(f"Expected {PATTERN_COUNT} values, got {len(values)}")
    return values


def load_baseline() -> list[float]:
    """Parse best/pattern_data.h and return the 729 values."""
    return load_weights(PATTERN_DATA_PATH)


def load_current() -> list[float]:
    """Parse current/pattern_data.h and return the 729 values."""
    return load_weights(CURRENT_PATTERN_DATA_PATH)


def save_pattern_data_h(full_params: list[float], path: str):
    """Write a pattern_data.h file from a full 729 parameter vector."""
    with open(path, "w") as f:
        f.write("#pragma once\n\n")
        f.write(f"static constexpr int PATTERN_EVAL_LENGTH = {PATTERN_LENGTH};\n")
        f.write(f"static constexpr int PATTERN_COUNT = {len(full_params)};\n")
        f.write("static const double PATTERN_VALUES[] = {\n")
        for i in range(0, len(full_params), 10):
            chunk = full_params[i : i + 10]
            f.write("    " + ", ".join(f"{v}" for v in chunk) + ",\n")
        f.write("};\n")
