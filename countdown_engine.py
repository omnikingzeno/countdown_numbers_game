"""Shared Countdown solver engine for CLI versions.

This module centralizes validation, candidate generation, deterministic
selection, and scoring so version-specific CLIs can stay thin.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations
from typing import Any


STANDARD_POOL_COUNTS = Counter(
    {
        25: 1,
        50: 1,
        75: 1,
        100: 1,
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
    }
)


def parse_target(raw: str) -> int:
    """Parse and validate the target number."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Target is required.")

    try:
        target = int(cleaned)
    except ValueError as exc:
        raise ValueError("Target must be an integer in the range 100 to 999.") from exc

    if target < 100 or target > 999:
        raise ValueError("Target must be between 100 and 999 inclusive.")

    return target


def parse_pool_numbers(raw: str, n: int) -> tuple[int, ...]:
    """Parse n space-separated integers and validate them against the pool."""
    parts = raw.strip().split()
    if len(parts) != n:
        raise ValueError(f"Enter exactly {n} numbers separated by spaces.")

    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "All entries must be integers from the standard pool."
        ) from exc

    validate_pool_selection(numbers)
    return numbers


def validate_pool_selection(nums: tuple[int, ...]) -> None:
    """Validate selected numbers against standard pool membership and copy counts."""
    selected_counts = Counter(nums)
    for number, selected_count in selected_counts.items():
        allowed_count = STANDARD_POOL_COUNTS.get(number, 0)
        if allowed_count == 0:
            raise ValueError(
                f"{number} is not in the standard Countdown pool "
                "(allowed: 1-10, 25, 50, 75, 100)."
            )
        if selected_count > allowed_count:
            raise ValueError(
                f"{number} can be selected at most {allowed_count} time(s) "
                "from the standard pool."
            )


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


def generate_binary_results(
    x: int,
    y: int,
    x_expr: str,
    y_expr: str,
) -> list[dict[str, Any]]:
    """Generate legal binary operation outcomes in fixed order for oriented operands."""
    results: list[dict[str, Any]] = [
        {"expr": _format_binary_expr(x_expr, "+", y_expr), "value": x + y},
        {"expr": _format_binary_expr(x_expr, "*", y_expr), "value": x * y},
    ]

    sub_result = x - y
    if sub_result >= 0:
        results.append(
            {"expr": _format_binary_expr(x_expr, "-", y_expr), "value": sub_result}
        )

    if y != 0 and x % y == 0:
        results.append(
            {"expr": _format_binary_expr(x_expr, "/", y_expr), "value": x // y}
        )

    return results


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence of candidates in deterministic order."""
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for candidate in candidates:
        key = (candidate["expr"], candidate["value"], candidate["used_count"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped


def _iter_unique_partitions(
    values: tuple[int, ...],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate unique two-way partitions of a sorted values tuple."""
    count = len(values)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    partitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    for size in range(1, count):
        for left_indices in combinations(range(count), size):
            left_set = set(left_indices)
            left = tuple(values[index] for index in range(count) if index in left_set)
            right = tuple(
                values[index] for index in range(count) if index not in left_set
            )
            pair = (left, right) if left <= right else (right, left)
            if pair in seen:
                continue
            seen.add(pair)
            partitions.append(pair)

    return partitions


def _maybe_update_best_for_value(
    best_by_value: dict[int, dict[str, Any]],
    value: int,
    expr: str,
    depth: int,
    order: int,
) -> None:
    """Store best expression for a value using (depth, order) ranking."""
    existing = best_by_value.get(value)
    if existing is None or depth < existing["operation_depth"]:
        best_by_value[value] = {
            "expr": expr,
            "operation_depth": depth,
            "order_index": order,
        }


@lru_cache(maxsize=None)
def _generate_exact_value_map(
    values: tuple[int, ...],
) -> tuple[tuple[int, str, int, int], ...]:
    """Return best exact candidates per value for this multiset of numbers.

    Each entry is: (value, expr, operation_depth, order_index).
    """
    if len(values) == 1:
        value = values[0]
        return ((value, str(value), 0, 0),)

    best_by_value: dict[int, dict[str, Any]] = {}
    order_counter = 0

    for left_values, right_values in _iter_unique_partitions(values):
        left_entries = _generate_exact_value_map(left_values)
        right_entries = _generate_exact_value_map(right_values)

        for left_value, left_expr, left_depth, _ in left_entries:
            for right_value, right_expr, right_depth, _ in right_entries:
                depth = max(left_depth, right_depth) + 1

                # Addition (commutative) - one direction is enough.
                add_expr = _format_binary_expr(left_expr, "+", right_expr)
                _maybe_update_best_for_value(
                    best_by_value,
                    left_value + right_value,
                    add_expr,
                    depth,
                    order_counter,
                )
                order_counter += 1

                # Multiplication (commutative) with an obvious prune from article.
                if left_value != 1 and right_value != 1:
                    mul_expr = _format_binary_expr(left_expr, "*", right_expr)
                    _maybe_update_best_for_value(
                        best_by_value,
                        left_value * right_value,
                        mul_expr,
                        depth,
                        order_counter,
                    )
                order_counter += 1

                # Subtraction (both directions) with non-negative constraint.
                sub_lr = left_value - right_value
                if sub_lr >= 0:
                    sub_lr_expr = _format_binary_expr(left_expr, "-", right_expr)
                    _maybe_update_best_for_value(
                        best_by_value,
                        sub_lr,
                        sub_lr_expr,
                        depth,
                        order_counter,
                    )
                order_counter += 1

                sub_rl = right_value - left_value
                if sub_rl >= 0:
                    sub_rl_expr = _format_binary_expr(right_expr, "-", left_expr)
                    _maybe_update_best_for_value(
                        best_by_value,
                        sub_rl,
                        sub_rl_expr,
                        depth,
                        order_counter,
                    )
                order_counter += 1

                # Division (both directions) with integer/intermediate constraints.
                if right_value not in {0, 1} and left_value % right_value == 0:
                    div_lr_expr = _format_binary_expr(left_expr, "/", right_expr)
                    _maybe_update_best_for_value(
                        best_by_value,
                        left_value // right_value,
                        div_lr_expr,
                        depth,
                        order_counter,
                    )
                order_counter += 1

                if left_value not in {0, 1} and right_value % left_value == 0:
                    div_rl_expr = _format_binary_expr(right_expr, "/", left_expr)
                    _maybe_update_best_for_value(
                        best_by_value,
                        right_value // left_value,
                        div_rl_expr,
                        depth,
                        order_counter,
                    )
                order_counter += 1

    as_list = [
        (value, info["expr"], info["operation_depth"], info["order_index"])
        for value, info in best_by_value.items()
    ]
    as_list.sort(key=lambda item: item[3])
    return tuple(as_list)


def generate_candidates_for_subset(nums: tuple[int, ...]) -> list[dict[str, Any]]:
    """Generate legal candidates that use all values in nums exactly once."""
    if not nums:
        raise ValueError("Subset cannot be empty.")

    sorted_values = tuple(sorted(nums))
    exact_entries = _generate_exact_value_map(sorted_values)
    used_count = len(nums)

    return [
        {
            "expr": expr,
            "value": value,
            "used_count": used_count,
            "operation_depth": depth,
            "order_index": order_index,
        }
        for value, expr, depth, order_index in exact_entries
    ]


def generate_all_candidates(nums: tuple[int, ...]) -> list[dict[str, Any]]:
    """Generate all legal candidates across subset sizes 1..len(nums)."""
    if not nums:
        raise ValueError("At least one number is required.")

    generated: list[dict[str, Any]] = []
    total = len(nums)

    for subset_size in range(1, total + 1):
        seen_subsets: set[tuple[int, ...]] = set()
        for index_group in combinations(range(total), subset_size):
            subset = tuple(nums[index] for index in index_group)
            if subset in seen_subsets:
                continue
            seen_subsets.add(subset)
            generated.extend(generate_candidates_for_subset(subset))

    deduped = _dedupe_candidates(generated)
    for index, candidate in enumerate(deduped):
        candidate["order_index"] = index

    return deduped


def find_best_candidate(nums: tuple[int, ...], target: int) -> dict[str, Any]:
    """Find best candidate without materializing every candidate globally."""
    if not nums:
        raise ValueError("At least one number is required.")

    total = len(nums)
    best: dict[str, Any] | None = None
    order_index = 0

    for subset_size in range(1, total + 1):
        seen_subsets: set[tuple[int, ...]] = set()
        for index_group in combinations(range(total), subset_size):
            subset = tuple(nums[index] for index in index_group)
            if subset in seen_subsets:
                continue
            seen_subsets.add(subset)

            for candidate in generate_candidates_for_subset(subset):
                candidate_with_order = {
                    "expr": candidate["expr"],
                    "value": candidate["value"],
                    "used_count": candidate["used_count"],
                    "operation_depth": candidate.get("operation_depth", 0),
                    "order_index": order_index,
                }
                order_index += 1

                if best is None:
                    best = candidate_with_order
                    continue

                best_key = (
                    abs(best["value"] - target),
                    -best["used_count"],
                    best.get("operation_depth", 0),
                    best.get("order_index", 0),
                )
                candidate_key = (
                    abs(candidate_with_order["value"] - target),
                    -candidate_with_order["used_count"],
                    candidate_with_order.get("operation_depth", 0),
                    candidate_with_order.get("order_index", 0),
                )
                if candidate_key < best_key:
                    best = candidate_with_order

    if best is None:
        raise ValueError("No valid candidates were generated.")

    return best


def choose_best(candidates: list[dict[str, Any]], target: int) -> dict[str, Any]:
    """Select the best candidate using deterministic tie-breaking."""
    if not candidates:
        raise ValueError("No valid candidates were generated.")

    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate["value"] - target),
            -candidate["used_count"],
            candidate.get("operation_depth", 0),
            candidate.get("order_index", 0),
        ),
    )


def score_points(distance: int) -> int:
    """Score according to standard Countdown thresholds."""
    if distance == 0:
        return 10
    if distance <= 5:
        return 7
    if distance <= 10:
        return 5
    return 0


def solve(target: int, nums: tuple[int, ...]) -> dict[str, Any]:
    """Solve a target for a validated tuple of numbers."""
    best = find_best_candidate(nums, target)

    distance = abs(best["value"] - target)
    return {
        "target": target,
        "numbers": nums,
        "best_expression": best["expr"],
        "value": best["value"],
        "distance": distance,
        "exact": distance == 0,
        "points": score_points(distance),
        "used_count": best["used_count"],
        "operation_depth": best.get("operation_depth", 0),
    }
