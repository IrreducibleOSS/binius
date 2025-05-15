#!/usr/bin/env python3
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
import argparse
from pathlib import Path
import subprocess
import os
import sys
import csv
import re
import time
import json

#
# Benchmark Config
#


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for a single benchmark.
    """
    name: str
    display: str
    args: List[str]
    n_ops: int
    single_threaded: bool = False
    exponent: int = field(init=False)

    def __post_init__(self):
        # Compute strict log2 and verify n_ops is a power of two
        x = self.n_ops
        if x <= 0 or (x & (x - 1)) != 0:
            raise ValueError(f"n_ops must be a positive power of 2, got {x}")
        # bit_length - 1 gives log2 for powers of two
        object.__setattr__(self, 'exponent', x.bit_length() - 1)


# Note: Every benchmark is multi-threaded by default. On top of that it can be run in single-threaded mode.
#  ┌──── name ───┬────── display ───────┬───────────────────── args ─────────────────────┬─ n_ops ─┬─ single_threaded ─┐
_RAW_BENCH_ROWS = [
    ("keccakf",   "Keccak-f",           ["keccak", "--", "--n-permutations"],                 1 << 13, True),
    ("groestl",   "Grøstl-256",         ["groestl", "--", "--n-permutations"],                1 << 14, True),
    ("vision32b", "Vision Mark-32",     ["vision32b_circuit", "--", "--n-permutations"],      1 << 14, True),
    ("sha256",    "SHA-256",            ["sha256_circuit", "--", "--n-compressions"],         1 << 14, True),
    ("b32_mul",   "BinaryField32b mul", ["b32_mul", "--", "--n-ops"],                         1 << 20, False),
    ("u32_add",   "u32 add",            ["u32_add", "--", "--n-additions"],                   1 << 22, False),
    ("u32_mul",   "u32 mul",            ["u32_mul", "--", "--n-muls"],                        1 << 20, False),
    ("xor",       "XOR (32-bit)",       ["bitwise_ops", "--", "--op", "xor",  "--n-u32-ops"], 1 << 22, False),
    ("and",       "AND (32-bit)",       ["bitwise_ops", "--", "--op", "and",  "--n-u32-ops"], 1 << 22, False),
    ("or",        "OR (32-bit)",        ["bitwise_ops", "--", "--op", "or",   "--n-u32-ops"], 1 << 22, False),
]

BENCHMARKS = {
    name: BenchmarkConfig(name, display, args, n_ops, single_threaded)
    for name, display, args, n_ops, single_threaded in _RAW_BENCH_ROWS
}


#
# Parse Script Arguments
#


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one benchmark example multiple times"
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to write CSVs & traces",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5,
        help="How many times to repeat the benchmark",
    )
    parser.add_argument(
        "--perfetto/--no-perfetto",
        dest="perfetto",
        default=True,
        help="Whether to generate Perfetto traces",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous results for selected benchmarks before running",
    )
    return parser.parse_args()


#
# Parse & Aggregate Results
#


def nano_to_milli(nano: int) -> float:
    return nano / 1e6


LABEL_TO_METRIC = {
    "generating trace":          "trace-gen-time",
    "constraint_system::prove":  "prove-time",
    "constraint_system::verify": "verify-time",
}


def parse_csv_metrics(csv_path: Path, key_suffix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            label = row[0]
            if label in LABEL_TO_METRIC:
                key = LABEL_TO_METRIC[label]
                metrics[key] = nano_to_milli(int(row[2]))
    missing = set(LABEL_TO_METRIC.values()) - metrics.keys()
    if missing:
        raise RuntimeError(f"Missing metrics in '{csv_path.name}': {', '.join(missing)}")
    return {k + '-' + key_suffix: v for k, v in metrics.items()}


SIZE_CONVERSIONS = {
    "B":   1,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "KB":  1000,
    "MB":  1000**2,
    "GB":  1000**3,
}


def parse_proof_size_kb(output: str) -> int:
    """
    Extract 'Proof size: <value><unit>' from stdout and convert to KiB.
    """
    match = re.search(r"Proof size: ([\d\.]+)\s*([A-Za-z]+)", output)
    if not match:
        raise RuntimeError("Failed to extract proof size from output. Missing 'Proof size: <value><unit>'")
    val_str, unit_str = match.groups()
    unit = unit_str.upper()
    factor = SIZE_CONVERSIONS.get(unit)
    if factor is None:
        raise ValueError(f"Unknown proof size unit: '{unit_str}'. Allowed units: {', '.join(SIZE_CONVERSIONS.keys())}")
    bytes_len = float(val_str) * factor
    return int(bytes_len // 1024)  # Convert to KiB

#
# Store Results
#


def write_results_to_json(
    display: str,
    n_ops: int,
    multi_data: Dict[str, list],
    single_data: Dict[str, list],
    json_path: Path
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"display": display, "n_ops": n_ops}
    for k, v in multi_data.items():
        result[k] = list(v)  # Copy the list to avoid modifying the original
    if single_data:
        for k, v in single_data.items():
            result.setdefault(k, []).extend(v)
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return result


#
# Execute a Benchmark
#


def run_single_benchmark(
    name: str,
    args: List[str],
    n_ops: int,
    outdir: Path,
    samples: int = 5,
    perfetto: bool = True,
    multi_threaded: bool = True,
) -> Dict[str, List[Any]]:
    """
    Run one benchmark `samples` times.
    Returns a dict mapping metric keys to lists of values.
    """

    result: Dict[str, List[Any]] = {}

    mode = 'multi-thread' if multi_threaded else 'single-thread'

    env = {
        **os.environ,
        'RAYON_NUM_THREADS': '0' if multi_threaded else '1',
        'RUSTFLAGS': '-C target-cpu=native',
    }

    cmd = ['cargo', 'run', '--release', '--example'] + args + [str(n_ops)]

    if perfetto:
        env['PERFETTO_TRACE_DIR'] = str(outdir)
        # Insert 'perfetto' feature args at index 3
        cmd[3:3] = ['--features', 'perfetto']

    # File containing the path to the last perfetto trace
    trace_pointer = Path('.last_perfetto_trace_path')

    for i in range(1, samples + 1):
        # Prepare for next iteration
        trace_pointer.unlink(missing_ok=True)

        csv_result_path = outdir / f"{name}-{mode}-{i}.csv"
        env['PROFILE_CSV_FILE'] = str(csv_result_path)
        print(f"Running {name} ({mode}) sample #{i}")
        print(f"{' '.join(cmd)}")

        # Run the benchmark
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Print a “heartbeat” while it’s running
        last_dot = 0
        while proc.poll() is None:
            time.sleep(0.5)
            if time.monotonic() - last_dot >= 5:
                print('.', end='', flush=True)
                last_dot = time.monotonic()

        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            print(" failed")
            print(stdout, end='')
            print(stderr, file=sys.stderr, end='')
            sys.exit(proc.returncode)
        else:
            print(" done")

        # Post-run: Collect results and clean up
        if perfetto:
            # Rename Perfetto trace file
            if not trace_pointer.exists():
                raise RuntimeError("Perfetto trace file not found (missing .last_perfetto_trace_path)")
            trace_path = Path(trace_pointer.read_text().strip())
            if not trace_path.exists():
                raise RuntimeError(f"Perfetto trace file not found: {trace_path} (read from .last_perfetto_trace_path)")
            trace_pointer.unlink()

            new_path = trace_path.parent / f"{name}-{mode}-{i}-{trace_path.name}"
            trace_path.rename(new_path)

        if not csv_result_path.exists():
            raise RuntimeError(f"CSV result file not found: {csv_result_path}")

        metrics = parse_csv_metrics(csv_result_path, mode)
        for k, v in metrics.items():
            result.setdefault(k, []).append(v)
        proof_size_kb = parse_proof_size_kb(stdout)
        result.setdefault("proof-size", []).append(proof_size_kb)

    return result

#
# Main
#


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine benchmarks to run
    if args.benchmark == "all":
        run_names = list(BENCHMARKS.keys())
    else:
        run_names = [args.benchmark]

    # Clean previous result files if requested
    if args.clean:
        for benchmark_name in run_names:
            prefix = f"{benchmark_name}-"
            for f in args.output_dir.glob(f"{prefix}*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    all_results = {}
    # Run each benchmark and write its results
    for benchmark_name in run_names:
        cfg = BENCHMARKS[benchmark_name]

        # Always run multi-threaded
        mt_data = run_single_benchmark(
            name=cfg.name,
            args=cfg.args,
            n_ops=cfg.n_ops,
            outdir=args.output_dir,
            samples=args.samples,
            perfetto=args.perfetto,
            multi_threaded=True,
        )
        # Then, if configured, run single-threaded too
        st_data: Optional[Dict[str, List[Any]]] = None
        if cfg.single_threaded:
            st_data = run_single_benchmark(
                name=cfg.name,
                args=cfg.args,
                n_ops=cfg.n_ops,
                outdir=args.output_dir,
                samples=args.samples,
                perfetto=args.perfetto,
                multi_threaded=False,
            )

        all_results[benchmark_name] = write_results_to_json(
            display=cfg.display,
            n_ops=cfg.n_ops,
            multi_data=mt_data,
            single_data=st_data,
            json_path=args.output_dir / f"{benchmark_name}-results.json",
        )

    (args.output_dir / "all-results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


if __name__ == "__main__":
    main()
