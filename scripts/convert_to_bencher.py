#!/usr/bin/env python3
from typing import Union, Dict, Any
import json
import argparse
from pathlib import Path


def strict_log2(x: int) -> int:
    """
    Calculate the strict log base 2 of an integer.
    Raises ValueError if x is not a power of 2.
    """
    if x <= 0 or not (x & (x - 1) == 0):  # Check if x is power of 2
        raise ValueError(f"{x} is not a positive power of 2")
    return x.bit_length() - 1


def value_to_bencher(metric_name: str, value: Union[list[float], int]) -> dict:
    if not isinstance(value, list):
        return {metric_name: {"value": value}}
    return {
        metric_name: {
            "value": sum(value) / len(value),
            "upper_value": max(value),
            "lower_value": min(value),
        }
    }


METRIC_BASE_KEYS = [
    "trace-gen-time",
    "prove-time",
    "verify-time",
    "phase-commit-time",
    "phase-zerocheck-time",
    "phase-evalcheck-time",
    "phase-ring-switch-time",
    "phase-piop-compiler-time",
]

METRIC_KEYS = [
    "proof-size",
] + [
    f"{base}-{thread}"
    for base in METRIC_BASE_KEYS
    for thread in ["multi-thread", "single-thread"]
]


def dict_to_bencher(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for record in data.values():
        # Name is of the following format: E2E / <benchmark_name> / n_ops
        common_name = f"E2E / {record['display']} / 2^{strict_log2(record['n_ops'])}"

        result.setdefault(common_name, {})
        for key in METRIC_KEYS:
            value = record.get(key)
            if value is None:
                continue
            result[common_name].update(value_to_bencher(key, value))
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Convert raw benchmark results JSON (multiple runs per benchmark) '
            'into a Bencher-compatible format, computing min, max, and average '
            'for each benchmark.'
        )
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to the raw benchmark results JSON file (contains multiple runs per benchmark).'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Path where the Bencher-compatible JSON file will be written.'
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        parser.error(f"Input file {args.input_file} does not exist.")

    raw_data = json.loads(args.input_file.read_text())

    bencher_data = dict_to_bencher(raw_data)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    args.output_file.write_text(
        json.dumps(bencher_data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    print(f"Wrote {len(bencher_data)} benchmarks to {args.output_file}")


if __name__ == '__main__':
    main()
