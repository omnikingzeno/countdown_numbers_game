"""Benchmark helper for Countdown solver.

This script measures solver runtime on a deterministic workload so you can
record a baseline now and compare future optimizations against the same cases.

Examples:
    python benchmark.py
    python benchmark.py --number-count 4 --cases 300 --repeats 7 --seed 42
    python benchmark.py --save-cases benchmarks/v5_cases.json
    python benchmark.py --cases-file benchmarks/v5_cases.json --output benchmarks/v5_baseline.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import random
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import cast


@dataclass(frozen=True)
class PuzzleCase:
    """Single benchmark case for the solver."""

    target: int
    numbers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "numbers": list(self.numbers)}


@dataclass(frozen=True)
class EngineRuntime:
    """Loaded solver module and required callables/constants."""

    module_name: str
    solve: Callable[[int, tuple[int, ...]], dict[str, Any]]
    validate_pool_selection: Callable[[tuple[int, ...]], None]
    pool_counts: dict[int, int]


def load_engine(module_name: str) -> EngineRuntime:
    """Load an engine module by name and validate required API."""
    module = importlib.import_module(module_name)

    required = ["solve", "validate_pool_selection", "STANDARD_POOL_COUNTS"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Engine module '{module_name}' missing required attributes: {missing_list}."
        )

    solve_func = getattr(module, "solve")
    validate_func = getattr(module, "validate_pool_selection")
    pool_counts_obj = getattr(module, "STANDARD_POOL_COUNTS")

    if not callable(solve_func):
        raise ValueError(f"Engine module '{module_name}' has non-callable solve.")
    if not callable(validate_func):
        raise ValueError(
            f"Engine module '{module_name}' has non-callable validate_pool_selection."
        )

    try:
        pool_counts = dict(pool_counts_obj)
    except TypeError as exc:
        raise ValueError(
            f"Engine module '{module_name}' has invalid STANDARD_POOL_COUNTS mapping."
        ) from exc

    typed_solve = cast(
        Callable[[int, tuple[int, ...]], dict[str, Any]],
        solve_func,
    )
    typed_validate = cast(Callable[[tuple[int, ...]], None], validate_func)

    return EngineRuntime(
        module_name=module_name,
        solve=typed_solve,
        validate_pool_selection=typed_validate,
        pool_counts=pool_counts,
    )


def build_pool_tiles(pool_counts: dict[int, int]) -> tuple[int, ...]:
    """Expand pool counts into a stable tuple for random sampling."""
    tiles: list[int] = []
    for value, copies in pool_counts.items():
        tiles.extend([value] * copies)
    return tuple(sorted(tiles))


def validate_case(
    case: PuzzleCase,
    index: int,
    number_count: int,
    validate_pool_selection_func: Callable[[tuple[int, ...]], None],
) -> None:
    """Validate target and number selection constraints for one case."""
    if case.target < 100 or case.target > 999:
        raise ValueError(
            f"Case {index} has target {case.target}, expected range 100..999."
        )
    if len(case.numbers) != number_count:
        raise ValueError(
            f"Case {index} has {len(case.numbers)} numbers, expected {number_count}."
        )
    validate_pool_selection_func(case.numbers)


def generate_cases(
    case_count: int,
    seed: int,
    number_count: int,
    pool_counts: dict[int, int],
) -> list[PuzzleCase]:
    """Generate deterministic random cases from the Countdown pool."""
    if case_count <= 0:
        raise ValueError("--cases must be >= 1.")

    rng = random.Random(seed)
    pool_tiles = build_pool_tiles(pool_counts)

    cases: list[PuzzleCase] = []
    for _ in range(case_count):
        draw = rng.sample(pool_tiles, number_count)
        case = PuzzleCase(
            target=rng.randint(100, 999),
            numbers=tuple(draw),
        )
        cases.append(case)

    return cases


def save_cases(path: Path, cases: list[PuzzleCase]) -> None:
    """Persist generated benchmark workload to JSON."""
    payload = [case.to_dict() for case in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cases(path: Path, number_count: int) -> list[PuzzleCase]:
    """Load benchmark workload from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Cases file must contain a JSON array.")

    cases: list[PuzzleCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} must be a JSON object.")

        target = item.get("target")
        numbers = item.get("numbers")

        if not isinstance(target, int):
            raise ValueError(f"Case {index} target must be an integer.")
        if not isinstance(numbers, list) or len(numbers) != number_count:
            raise ValueError(
                "Case "
                f"{index} numbers must be a list of exactly {number_count} integers."
            )
        if not all(isinstance(number, int) for number in numbers):
            raise ValueError(f"Case {index} numbers must contain only integers.")

        case = PuzzleCase(
            target=target,
            numbers=tuple(numbers),
        )
        cases.append(case)

    if not cases:
        raise ValueError("Cases file is empty.")

    return cases


def run_warmup(
    cases: list[PuzzleCase],
    rounds: int,
    solve_func: Callable[[int, tuple[int, ...]], dict[str, Any]],
) -> None:
    """Run unmeasured solves to warm caches and interpreter paths."""
    if rounds < 0:
        raise ValueError("--warmup must be >= 0.")

    for _ in range(rounds):
        for case in cases:
            solve_func(case.target, case.numbers)


def run_benchmark(
    cases: list[PuzzleCase],
    repeats: int,
    check_determinism: bool,
    solve_func: Callable[[int, tuple[int, ...]], dict[str, Any]],
) -> list[float]:
    """Run benchmark repeats and return per-repeat elapsed seconds."""
    if repeats <= 0:
        raise ValueError("--repeats must be >= 1.")

    per_repeat_seconds: list[float] = []
    baseline_outputs: list[tuple[str, int, int, int]] | None = None

    for _ in range(repeats):
        repeat_outputs: list[tuple[str, int, int, int]] = []
        started = perf_counter()

        for case in cases:
            result = solve_func(case.target, case.numbers)
            if check_determinism:
                repeat_outputs.append(
                    (
                        result["best_expression"],
                        int(result["value"]),
                        int(result["distance"]),
                        int(result["used_count"]),
                    )
                )

        elapsed = perf_counter() - started
        per_repeat_seconds.append(elapsed)

        if check_determinism:
            if baseline_outputs is None:
                baseline_outputs = repeat_outputs
            elif repeat_outputs != baseline_outputs:
                raise RuntimeError(
                    "Solver outputs changed across repeats; benchmark is not deterministic."
                )

    return per_repeat_seconds


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation percentile for p in [0, 1]."""
    if not values:
        raise ValueError("Cannot compute percentile of empty list.")
    if p < 0.0 or p > 1.0:
        raise ValueError("Percentile p must be in [0, 1].")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = p * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower

    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize(
    per_repeat_seconds: list[float], cases_per_repeat: int
) -> dict[str, float]:
    """Build summary metrics in both repeat and per-solve units."""
    per_solve_seconds = [elapsed / cases_per_repeat for elapsed in per_repeat_seconds]

    stdev_repeat = (
        statistics.stdev(per_repeat_seconds) if len(per_repeat_seconds) > 1 else 0.0
    )
    stdev_solve = (
        statistics.stdev(per_solve_seconds) if len(per_solve_seconds) > 1 else 0.0
    )

    return {
        "repeat_total_seconds_mean": statistics.mean(per_repeat_seconds),
        "repeat_total_seconds_median": statistics.median(per_repeat_seconds),
        "repeat_total_seconds_p95": percentile(per_repeat_seconds, 0.95),
        "repeat_total_seconds_min": min(per_repeat_seconds),
        "repeat_total_seconds_max": max(per_repeat_seconds),
        "repeat_total_seconds_stdev": stdev_repeat,
        "per_solve_ms_mean": statistics.mean(per_solve_seconds) * 1000.0,
        "per_solve_ms_median": statistics.median(per_solve_seconds) * 1000.0,
        "per_solve_ms_p95": percentile(per_solve_seconds, 0.95) * 1000.0,
        "per_solve_ms_min": min(per_solve_seconds) * 1000.0,
        "per_solve_ms_max": max(per_solve_seconds) * 1000.0,
        "per_solve_ms_stdev": stdev_solve * 1000.0,
    }


def build_report(
    engine_module: str,
    number_count: int,
    cases: list[PuzzleCase],
    per_repeat_seconds: list[float],
    summary: dict[str, float],
    repeats: int,
    warmup: int,
    workload: str,
    seed: int | None,
) -> dict[str, Any]:
    """Build JSON report payload for baseline storage."""
    return {
        "benchmark": "countdown-solver",
        "engine_module": engine_module,
        "number_count": number_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "solver_module": f"{engine_module}.solve",
        "workload": workload,
        "seed": seed,
        "cases": len(cases),
        "repeats": repeats,
        "warmup_rounds": warmup,
        "total_measured_solves": len(cases) * repeats,
        "per_repeat_seconds": per_repeat_seconds,
        "summary": summary,
    }


def print_report(
    report: dict[str, Any],
) -> None:
    """Print human-readable benchmark output."""
    summary = report["summary"]

    print("Countdown benchmark")
    print(f"Engine: {report['engine_module']}")
    print(f"Number count: {report['number_count']}")
    print(f"Python: {report['python_version']}")
    print(f"Platform: {report['platform']}")
    print(f"Workload: {report['workload']}")
    if report["seed"] is not None:
        print(f"Seed: {report['seed']}")
    print(f"Cases per repeat: {report['cases']}")
    print(f"Repeats: {report['repeats']}")
    print(f"Warmup rounds: {report['warmup_rounds']}")
    print(f"Measured solves: {report['total_measured_solves']}")

    repeats_rendered = ", ".join(
        f"{value:.6f}" for value in report["per_repeat_seconds"]
    )
    print(f"Per-repeat seconds: {repeats_rendered}")

    print("Summary")
    print(f"  Mean repeat total: {summary['repeat_total_seconds_mean']:.6f} s")
    print(f"  Median repeat total: {summary['repeat_total_seconds_median']:.6f} s")
    print(f"  P95 repeat total: {summary['repeat_total_seconds_p95']:.6f} s")
    print(f"  Min repeat total: {summary['repeat_total_seconds_min']:.6f} s")
    print(f"  Max repeat total: {summary['repeat_total_seconds_max']:.6f} s")
    print(f"  Repeat stdev: {summary['repeat_total_seconds_stdev']:.6f} s")
    print(f"  Mean per solve: {summary['per_solve_ms_mean']:.3f} ms")
    print(f"  Median per solve: {summary['per_solve_ms_median']:.3f} ms")
    print(f"  P95 per solve: {summary['per_solve_ms_p95']:.3f} ms")
    print(f"  Min per solve: {summary['per_solve_ms_min']:.3f} ms")
    print(f"  Max per solve: {summary['per_solve_ms_max']:.3f} ms")
    print(f"  Per solve stdev: {summary['per_solve_ms_stdev']:.3f} ms")


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write benchmark report JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_comparison(
    base_report: dict[str, Any],
    optimized_report: dict[str, Any],
) -> dict[str, float]:
    """Print side-by-side comparison and return comparison metrics."""
    base_ms = float(base_report["summary"]["per_solve_ms_mean"])
    optimized_ms = float(optimized_report["summary"]["per_solve_ms_mean"])

    if optimized_ms == 0.0:
        speedup = float("inf")
    else:
        speedup = base_ms / optimized_ms

    delta_ms = base_ms - optimized_ms
    percent_change = 0.0 if base_ms == 0.0 else (delta_ms / base_ms) * 100.0

    print("Engine comparison")
    print(f"  {base_report['engine_module']} mean per solve: {base_ms:.3f} ms")
    print(
        f"  {optimized_report['engine_module']} mean per solve: {optimized_ms:.3f} ms"
    )
    print(f"  Delta: {delta_ms:.3f} ms")
    print(f"  Speedup: {speedup:.3f}x")
    print(f"  Percent improvement: {percent_change:.2f}%")

    return {
        "base_per_solve_ms_mean": base_ms,
        "optimized_per_solve_ms_mean": optimized_ms,
        "delta_per_solve_ms_mean": delta_ms,
        "speedup_factor": speedup,
        "percent_improvement": percent_change,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI options for benchmarking."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Countdown solver on deterministic workloads for 4..6 numbers."
        )
    )
    parser.add_argument(
        "--number-count",
        type=int,
        choices=[4, 5, 6],
        default=6,
        help="How many selected numbers each benchmark case should use (default: 6).",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=200,
        help="Number of generated benchmark cases per repeat (default: 200).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="How many measured repeats to run (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="How many unmeasured warmup repeats to run first (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260318,
        help="Seed used for generated workloads (default: 20260318).",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=None,
        help="Optional JSON workload file to load instead of generating cases.",
    )
    parser.add_argument(
        "--save-cases",
        type=Path,
        default=None,
        help="Save generated workload to this JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report.",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="Skip validation that outputs remain identical across repeats.",
    )
    parser.add_argument(
        "--engine-module",
        type=str,
        default="countdown_engine",
        help="Primary engine module to benchmark (default: countdown_engine).",
    )
    parser.add_argument(
        "--compare-engine-module",
        type=str,
        default=None,
        help="Optional second engine module for side-by-side comparison.",
    )
    return parser.parse_args()


def main() -> int:
    """Run benchmark script."""
    args = parse_args()
    primary_engine = load_engine(args.engine_module)

    compare_engine: EngineRuntime | None = None
    if args.compare_engine_module is not None:
        compare_engine = load_engine(args.compare_engine_module)

    if (
        compare_engine is not None
        and compare_engine.pool_counts != primary_engine.pool_counts
    ):
        raise ValueError(
            "Compared engines use different STANDARD_POOL_COUNTS; "
            "workload generation would not be comparable."
        )

    if args.cases_file is not None:
        cases = load_cases(args.cases_file, args.number_count)
        workload = f"file:{args.cases_file}"
        workload_seed: int | None = None
    else:
        cases = generate_cases(
            args.cases,
            args.seed,
            args.number_count,
            primary_engine.pool_counts,
        )
        workload = "generated"
        workload_seed = args.seed
        if args.save_cases is not None:
            save_cases(args.save_cases, cases)

    for index, case in enumerate(cases):
        validate_case(
            case,
            index,
            args.number_count,
            primary_engine.validate_pool_selection,
        )
        if compare_engine is not None:
            validate_case(
                case,
                index,
                args.number_count,
                compare_engine.validate_pool_selection,
            )

    def run_for_engine(engine: EngineRuntime) -> dict[str, Any]:
        run_warmup(cases, args.warmup, engine.solve)

        per_repeat_seconds = run_benchmark(
            cases=cases,
            repeats=args.repeats,
            check_determinism=not args.skip_determinism_check,
            solve_func=engine.solve,
        )

        summary = summarize(per_repeat_seconds, len(cases))
        return build_report(
            engine_module=engine.module_name,
            number_count=args.number_count,
            cases=cases,
            per_repeat_seconds=per_repeat_seconds,
            summary=summary,
            repeats=args.repeats,
            warmup=args.warmup,
            workload=workload,
            seed=workload_seed,
        )

    primary_report = run_for_engine(primary_engine)
    print_report(primary_report)

    output_payload: dict[str, Any] = primary_report

    if compare_engine is not None:
        secondary_report = run_for_engine(compare_engine)
        print("")
        print_report(secondary_report)
        print("")
        comparison = print_comparison(primary_report, secondary_report)
        output_payload = {
            "benchmark": "countdown-engine-compare",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "workload": workload,
            "seed": workload_seed,
            "number_count": args.number_count,
            "cases": len(cases),
            "repeats": args.repeats,
            "warmup_rounds": args.warmup,
            "reports": {
                primary_engine.module_name: primary_report,
                compare_engine.module_name: secondary_report,
            },
            "comparison": comparison,
        }

    if args.output is not None:
        write_report(args.output, output_payload)
        print(f"Saved JSON report to: {args.output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
