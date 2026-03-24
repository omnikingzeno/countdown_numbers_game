from __future__ import annotations

import argparse
import gzip
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure project-root imports work when this script is run from benchmarks/.
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from countdown_engine_optimized import STANDARD_POOL_COUNTS

TARGET_MIN = 100
TARGET_MAX = 999


def unique_multisets_of_size(n: int) -> list[tuple[int, ...]]:
    items = sorted(STANDARD_POOL_COUNTS.items())

    def _recurse(index: int, remaining: int) -> list[tuple[int, ...]]:
        if remaining == 0:
            return [()]
        if index == len(items):
            return []

        value, cap = items[index]
        out: list[tuple[int, ...]] = []
        for take in range(min(cap, remaining) + 1):
            prefix = (value,) * take
            for tail in _recurse(index + 1, remaining - take):
                out.append(prefix + tail)
        return out

    return _recurse(0, n)


def unique_subsets_of_draw(draw: tuple[int, ...]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []

    for subset_size in range(1, len(draw) + 1):
        for index_group in combinations(range(len(draw)), subset_size):
            subset = tuple(draw[index] for index in index_group)
            if subset in seen:
                continue
            seen.add(subset)
            out.append(subset)

    return out


def _is_atomic_expression(expr: str) -> bool:
    return expr.isdigit()


def _is_grouped_expression(expr: str) -> bool:
    return expr.startswith("(") and expr.endswith(")")


def _format_operand(expr: str) -> str:
    if _is_atomic_expression(expr) or _is_grouped_expression(expr):
        return expr
    return f"({expr})"


def _format_binary_expr(left_expr: str, op: str, right_expr: str) -> str:
    left = _format_operand(left_expr)
    right = _format_operand(right_expr)
    return f"{left} {op} {right}"


def expression_sets_for_subset(numbers: tuple[int, ...]) -> dict[int, set[str]]:
    """Return value -> set of distinct expression strings for this exact subset.

    Rules followed:
    - Only +, -, *, /
    - Intermediate values must stay non-negative integers
    - Division must be exact integer division
    - Commutative duplicates are reduced by unique partitioning (left<right)
    """
    if not numbers:
        raise ValueError("Subset must be non-empty.")

    size = len(numbers)
    full_mask = (1 << size) - 1
    memo: dict[int, dict[int, set[str]]] = {}

    def add_expr(out: dict[int, set[str]], value: int, expr: str) -> None:
        bucket = out.get(value)
        if bucket is None:
            out[value] = {expr}
            return
        bucket.add(expr)

    def solve_mask(mask: int) -> dict[int, set[str]]:
        cached = memo.get(mask)
        if cached is not None:
            return cached

        # Single number leaf.
        if mask & (mask - 1) == 0:
            index = mask.bit_length() - 1
            value = numbers[index]
            result = {value: {str(value)}}
            memo[mask] = result
            return result

        result: dict[int, set[str]] = {}
        submask = (mask - 1) & mask

        while submask:
            left_mask = submask
            right_mask = mask ^ left_mask

            # Unique unordered partitions; subtraction/division handle both orders.
            if right_mask and left_mask < right_mask:
                left_map = solve_mask(left_mask)
                right_map = solve_mask(right_mask)

                for left_value, left_exprs in left_map.items():
                    for right_value, right_exprs in right_map.items():
                        for left_expr in left_exprs:
                            for right_expr in right_exprs:
                                add_expr(
                                    result,
                                    left_value + right_value,
                                    _format_binary_expr(left_expr, "+", right_expr),
                                )

                                add_expr(
                                    result,
                                    left_value * right_value,
                                    _format_binary_expr(left_expr, "*", right_expr),
                                )

                                sub_lr = left_value - right_value
                                if sub_lr >= 0:
                                    add_expr(
                                        result,
                                        sub_lr,
                                        _format_binary_expr(left_expr, "-", right_expr),
                                    )

                                sub_rl = right_value - left_value
                                if sub_rl >= 0:
                                    add_expr(
                                        result,
                                        sub_rl,
                                        _format_binary_expr(right_expr, "-", left_expr),
                                    )

                                if right_value != 0 and left_value % right_value == 0:
                                    add_expr(
                                        result,
                                        left_value // right_value,
                                        _format_binary_expr(left_expr, "/", right_expr),
                                    )

                                if left_value != 0 and right_value % left_value == 0:
                                    add_expr(
                                        result,
                                        right_value // left_value,
                                        _format_binary_expr(right_expr, "/", left_expr),
                                    )

            submask = (submask - 1) & mask

        memo[mask] = result
        return result

    return solve_mask(full_mask)


def precompute_subset_target_count_maps(
    max_n: int,
) -> dict[tuple[int, ...], dict[int, int]]:
    """Precompute target -> expression-count maps for all unique subsets up to max_n."""
    subset_count_maps: dict[tuple[int, ...], dict[int, int]] = {}

    for subset_size in range(1, max_n + 1):
        subsets = unique_multisets_of_size(subset_size)
        print(f"precomputing subset_size={subset_size}, subsets={len(subsets)}")

        for index, subset in enumerate(subsets, start=1):
            expr_map = expression_sets_for_subset(subset)
            target_counts: dict[int, int] = {}

            for value, exprs in expr_map.items():
                if TARGET_MIN <= value <= TARGET_MAX:
                    target_counts[value] = len(exprs)

            subset_count_maps[subset] = target_counts

            if index % 300 == 0:
                print(f"  subset_size={subset_size} processed={index}/{len(subsets)}")

    return subset_count_maps


def build_payload_for_n(n: int, sparse: bool) -> dict[str, Any]:
    draws = unique_multisets_of_size(n)
    subset_count_maps = precompute_subset_target_count_maps(n)

    number_sets: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for draw_index, draw in enumerate(draws, start=1):
        target_counts: dict[int, int] = {
            target: 0 for target in range(TARGET_MIN, TARGET_MAX + 1)
        }
        subsets = unique_subsets_of_draw(draw)

        for subset in subsets:
            subset_target_counts = subset_count_maps[subset]
            for target, count in subset_target_counts.items():
                target_counts[target] += count

        key = " ".join(str(value) for value in draw)
        if sparse:
            targets_map = {
                str(target): {"count_of_solutions": count}
                for target, count in target_counts.items()
                if count > 0
            }
        else:
            targets_map = {
                str(target): {"count_of_solutions": count}
                for target, count in target_counts.items()
            }

        number_sets[key] = {"targets": targets_map}

        if draw_index % 200 == 0:
            print(f"processed draws: {draw_index}/{len(draws)}")

    payload: dict[str, Any] = {
        "meta": {
            "n": n,
            "target_range": [TARGET_MIN, TARGET_MAX],
            "targets_per_number_set": TARGET_MAX - TARGET_MIN + 1,
            "format": (
                "number_sets[number_set_key].targets[target].count_of_solutions"
            ),
            "number_set_key": (
                "Space-separated sorted multiset of values selected from Countdown pool"
            ),
            "count_definition": (
                "Count of distinct expression strings that evaluate exactly to the target "
                "using up to n numbers from the given number set, under Countdown integer and "
                "non-negative intermediate constraints."
            ),
            "sparse": sparse,
        },
        "number_sets": number_sets,
    }
    return payload


def write_json(path: Path, payload: dict[str, Any], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        content = json.dumps(payload, indent=2) + "\n"
    else:
        content = json.dumps(payload, separators=(",", ":")) + "\n"
    path.write_text(content, encoding="utf-8")


def write_gzip_json(path: Path) -> Path:
    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(
        compressed_path, "wb", compresslevel=9
    ) as target:
        target.write(source.read())
    return compressed_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-number-set per-target algebraically distinct expression counts."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Input number count n (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/target_solutions_n4.json"),
        help=("Output JSON path " "(default: benchmarks/target_solutions_n4.json)."),
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Store only targets with non-zero counts for each number set.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation (larger files).",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Also write a gzip-compressed copy as <output>.gz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n < 1 or args.n > 6:
        raise ValueError("n must be between 1 and 6.")

    if args.n > 4:
        print(
            "Warning: algebraically-distinct expression counting grows quickly; "
            "n>4 may take substantial time and memory."
        )

    started = perf_counter()
    payload = build_payload_for_n(args.n, sparse=args.sparse)

    write_json(args.output, payload, pretty=args.pretty)
    print(f"wrote json: {args.output}")

    if args.gzip:
        compressed = write_gzip_json(args.output)
        print(f"wrote gzip: {compressed}")

    elapsed = perf_counter() - started
    number_sets = payload["number_sets"]
    if not isinstance(number_sets, dict):
        raise RuntimeError("Unexpected payload structure: number_sets is not a dict.")
    set_count = len(number_sets)
    print(f"number_sets_exported: {set_count}")
    print(f"elapsed_seconds: {elapsed:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
