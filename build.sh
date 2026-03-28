#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for bot in current best; do
    echo "Cleaning $bot ..."
    rm -rf "$bot"/build "$bot"/*.so "$bot"/*.pyd "$bot"/*.egg-info

    echo "Building $bot ..."
    (cd "$bot" && ../.venv/bin/python setup.py build_ext --inplace)
done

echo "Done."
