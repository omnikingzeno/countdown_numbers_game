"""Unified Countdown numbers game CLI.

Single entrypoint that lets the player choose how many numbers to use
(4, 5, or 6) before each round, and choose whether the
target is entered manually or generated randomly.
"""

from __future__ import annotations

import random

# Baseline engine imports (uncomment to switch back).
# from countdown_engine import parse_pool_numbers
# from countdown_engine import parse_target as _parse_target
# from countdown_engine import score_points as _score_points
# from countdown_engine import solve as _solve

from countdown_engine_optimized import parse_pool_numbers
from countdown_engine_optimized import parse_target as _parse_target
from countdown_engine_optimized import score_points as _score_points
from countdown_engine_optimized import solve as _solve

ALLOWED_NUMBER_COUNTS = {4, 5, 6}


def parse_target(raw: str) -> int:
    """Parse and validate the target number."""
    return _parse_target(raw)


def parse_target_mode(raw: str) -> str:
    """Parse and validate target mode selection."""
    cleaned = raw.strip().lower()
    if not cleaned:
        raise ValueError("Target mode is required.")

    if cleaned in {"m", "manual"}:
        return "manual"
    if cleaned in {"r", "random"}:
        return "random"

    raise ValueError("Choose M for manual target or R for random target.")


def parse_number_count(raw: str) -> int:
    """Parse and validate number-count choice for the round."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Number count is required.")

    try:
        number_count = int(cleaned)
    except ValueError as exc:
        raise ValueError("Choose 4, 5, or 6.") from exc

    if number_count not in ALLOWED_NUMBER_COUNTS:
        raise ValueError("Choose 4, 5, or 6.")

    return number_count


def parse_n_pool_numbers(raw: str, number_count: int) -> tuple[int, ...]:
    """Parse number_count pool numbers from user input."""
    return parse_pool_numbers(raw, number_count)


def score_points(distance: int) -> int:
    """Score according to standard Countdown thresholds."""
    return _score_points(distance)


def format_numbers(numbers: tuple[int, ...]) -> str:
    """Render selected numbers as a single space-separated string."""
    return " ".join(str(number) for number in numbers)


def run_cli_loop() -> None:
    """Run the unified interactive CLI loop."""
    print("Countdown Numbers Solver - Unified CLI")
    print("Target range: 100 to 999")
    print("Choose number count each round: 4, 5, or 6")
    print("Choose target mode each round: manual or random")
    print("Large: 25, 50, 75, 100")
    print("Small: 1-10 (two copies each)")

    while True:
        while True:
            raw_count = input("\nHow many numbers this round? (4/5/6): ")
            try:
                number_count = parse_number_count(raw_count)
                break
            except ValueError as error:
                print(f"Invalid choice: {error}")

        while True:
            raw_target_mode = input("Target mode (M=manual, R=random): ")
            try:
                target_mode = parse_target_mode(raw_target_mode)
                break
            except ValueError as error:
                print(f"Invalid target mode: {error}")

        if target_mode == "manual":
            while True:
                raw_target = input("Enter target (100-999): ")
                try:
                    target = parse_target(raw_target)
                    break
                except ValueError as error:
                    print(f"Invalid target: {error}")
        else:
            target = random.randint(100, 999)
            print(f"Random target selected: {target}")

        while True:
            raw_numbers = input(
                f"Enter {number_count} pool numbers (space-separated): "
            )
            try:
                numbers = parse_n_pool_numbers(raw_numbers, number_count)
                break
            except ValueError as error:
                print(f"Invalid numbers: {error}")

        result = _solve(target, numbers)
        distance = int(result["distance"])
        points = int(result["points"])
        exact = "Yes" if result["exact"] else "No"

        print("\nResult")
        print(f"Number count: {number_count}")
        print(f"Target: {target}")
        print(f"Numbers: {format_numbers(numbers)}")
        print(f"Best expression: {result['best_expression']}")
        print(f"Value: {result['value']}")
        print(f"Distance: {distance}")
        print(f"Exact: {exact}")
        print(f"Points: {points}")

        again = input("\nSolve another puzzle? (Y/N): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Exiting Countdown Numbers Solver.")
            break


if __name__ == "__main__":
    run_cli_loop()
