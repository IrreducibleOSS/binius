#!/usr/bin/python3

import argparse
import csv
import json
import os
import re
import subprocess
from typing import Union

ENV_VARS = {
    "RUSTFLAGS": "-C target-cpu=native",
}

SAMPLE_SIZE = 5

KECCAKF_PERMS = 1 << 13
VISION32B_PERMS = 1 << 14
SHA256_PERMS = 1 << 14
NUM_BINARY_OPS = 1 << 22
NUM_MULS = 1 << 20

# Examples to run
# Every example is run in multi-threaded mode by default
# To include single-threaded as well, set the `single_threaded` flag to True

EXAMPLES_TO_RUN = {
    r"keccakf": {
        "single_threaded": True,
        "display": r"Keccak-f",
        "export": "keccakf-report.csv",
        "args": ["keccakf_circuit", "--", "--n-permutations"],
        "n_ops": KECCAKF_PERMS,
    },
    "vision32b": {
        "single_threaded": True,
        "display": r"Vision Mark-32",
        "export": "vision32b-report.csv",
        "args": ["vision32b_circuit", "--", "--n-permutations"],
        "n_ops": VISION32B_PERMS,
    },
    "sha256": {
        "single_threaded": True,
        "display": "SHA-256",
        "export": "sha256-report.csv",
        "args": ["sha256_circuit", "--", "--n-compressions"],
        "n_ops": SHA256_PERMS,
    },
    "b32_mul": {
        "single_threaded": False,
        "display": "BinaryField32b mul",
        "export": "b32-mul-report.csv",
        "args": ["b32_mul", "--", "--n-ops"],
        "n_ops": NUM_MULS,
    },
    "u32_add": {
        "single_threaded": False,
        "display": "u32 add",
        "export": "u32-add-report.csv",
        "args": ["u32_add", "--", "--n-additions"],
        "n_ops": NUM_BINARY_OPS,
    },
    "u32_mul": {
        "single_threaded": False,
        "display": "u32 mul",
        "export": "u32-mul-report.csv",
        "args": ["u32_mul", "--", "--n-muls"],
        "n_ops": NUM_MULS,
    },
    "xor": {
        "single_threaded": False,
        "display": "XOR (32-bit)",
        "export": "xor-report.csv",
        "args": ["bitwise_ops", "--", "--op", "xor", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
    "and": {
        "single_threaded": False,
        "display": "AND (32-bit)",
        "export": "and-report.csv",
        "args": ["bitwise_ops", "--", "--op", "and", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
    "or": {
        "single_threaded": False,
        "display": "OR (32-bit)",
        "export": "or-report.csv",
        "args": ["bitwise_ops", "--", "--op", "or", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
}

HASHER_BENCHMARKS = {}
BINARY_OPS_BENCHMARKS = {}


def run_benchmark(benchmark_args, include_single_threaded) -> tuple[bytes, bytes]:
    command = (
        ["cargo", "run", "--features", "perfetto", "--release", "--example"]
        + benchmark_args["args"]
        + [f"{benchmark_args['n_ops']}"]
    )
    env_vars_to_run = {
        **os.environ,
        **ENV_VARS,
        "PROFILE_CSV_FILE": benchmark_args["export"],
    }
    if include_single_threaded:
        env_vars_to_run["RAYON_NUM_THREADS"] = "1"
    else:
        env_vars_to_run["RAYON_NUM_THREADS"] = "0"
    process = subprocess.run(
        command, env=env_vars_to_run, capture_output=True, check=True
    )
    return process.stdout, process.stderr


def parse_csv_file(file_name) -> dict:
    data = {}
    with open(file_name) as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "generating trace":
                data.update({"trace-gen-time": nano_to_milli(int(row[2]))})
            elif row[0] == "constraint_system::prove":
                data.update({"prove-time": nano_to_milli(int(row[2]))})
            elif row[0] == "constraint_system::verify":
                data.update({"verify-time": nano_to_milli(int(row[2]))})
    return data


def strict_log2(x: int) -> int:
    """
    Calculate the strict log base 2 of an integer.
    Raises ValueError if x is not a power of 2.
    """
    if x <= 0 or not (x & (x - 1) == 0):  # Check if x is power of 2
        raise ValueError(f"{x} is not a positive power of 2")
    return x.bit_length() - 1


KIB_TO_BYTES = 1024.0
MIB_TO_BYTES = KIB_TO_BYTES * 1024.0
GIB_TO_BYTES = MIB_TO_BYTES * 1024.0
KB_TO_BYTES = 1000.0
MB_TO_BYTES = KB_TO_BYTES * 1000.0
GB_TO_BYTES = MB_TO_BYTES * 1000.0

SIZE_CONVERSIONS = {
    "KiB": KIB_TO_BYTES,
    "MiB": MIB_TO_BYTES,
    "GiB": GIB_TO_BYTES,
    " B": 1,
    "KB": KB_TO_BYTES,
    "MB": MB_TO_BYTES,
    "GB": GB_TO_BYTES,
}


def parse_proof_size(proof_size: bytes) -> int:
    proof_size = proof_size.decode("utf-8").strip()
    for unit, factor in SIZE_CONVERSIONS.items():
        if proof_size.endswith(unit):
            byte_len = float(proof_size[: -len(unit)]) * factor
            break
    else:
        raise ValueError(f"Unknown proof size format: {proof_size}")

    # Convert to KiB
    return int(byte_len / KIB_TO_BYTES)


def nano_to_milli(nano) -> int:
    return int(float(nano) / 1000000.0)


def nano_to_seconds(nano) -> float:
    return float(nano) / 1000000000.0


def run_and_parse_benchmark(
    benchmark, benchmark_args, include_single_threaded
) -> tuple[dict, int]:
    data = {}
    stdout = None
    print(
        f"Running benchmark {benchmark} {'single-threaded' if include_single_threaded else 'multi-threaded'} with {SAMPLE_SIZE} samples"
    )
    for _ in range(SAMPLE_SIZE):
        stdout, _stderr = run_benchmark(benchmark_args, include_single_threaded)
        result = parse_csv_file(benchmark_args["export"])
        # Parse the csv file
        if len(result.keys()) != 3:
            print(f"Failed to parse csv file for benchmark: {benchmark}")
            exit(1)

        # Append the results to the data
        for key, value in result.items():
            if data.get(key) is None:
                data[key] = []
            data[key].append(value)
    # Move perfetto trace to example specific file
    os.rename("tracing.perfetto-trace", f"examples/{benchmark}.perfetto-trace")
    # Get proof sizes
    found = re.search(rb"Proof size: (.*)", stdout)
    if found:
        return data, parse_proof_size(found.group(1))
    else:
        print(f"Failed to get proof size for benchmark: {benchmark}")
        exit(1)


def run_benchmark_group(benchmarks) -> dict:
    benchmark_results = {}
    for benchmark, benchmark_args in benchmarks.items():
        try:
            data, proof_size = run_and_parse_benchmark(benchmark, benchmark_args, False)
            benchmark_results[benchmark] = {"proof-size": proof_size}
            multi_threaded_data = {}
            for key, value in data.items():
                multi_threaded_data[key + "-multi-thread"] = value
            multi_threaded_data["n_ops"] = benchmark_args["n_ops"]
            multi_threaded_data["display"] = benchmark_args["display"]
            benchmark_results[benchmark].update(multi_threaded_data)

            if benchmark_args["single_threaded"]:
                data, _proof_size = run_and_parse_benchmark(
                    benchmark, benchmark_args, True
                )
                single_threaded_data = {}
                for key, value in data.items():
                    single_threaded_data[key + "-single-thread"] = value
                benchmark_results[benchmark].update(single_threaded_data)

        except Exception as e:
            print(f"Failed to run benchmark: {benchmark} with error {e} \nExiting...")
            exit(1)
    return benchmark_results


def value_to_bencher(value: Union[list[float], int], metric_type: str) -> dict:
    if isinstance(value, list):
        avg_value = sum(value) / len(value)
        max_value = max(value)
        min_value = min(value)
        return {
            metric_type: {
                "value": avg_value,
                "upper_value": max_value,
                "lower_value": min_value,
            }
        }
    else:
        return {metric_type: {"value": value}}


def dict_to_bencher(data: dict) -> dict:
    bencher_data = {}
    for benchmark, value in data.items():
        # Name is of the following format: E2E / <benchmark_name> / n_ops
        common_name = f"E2E / {value['display']} / 2^{strict_log2(value['n_ops'])}"
        if bencher_data.get(common_name) is None:
            bencher_data[common_name] = {}
        for key in [
            "trace-gen-time-multi-thread",
            "trace-gen-time-single-thread",
            "prove-time-multi-thread",
            "prove-time-single-thread",
            "verify-time-multi-thread",
            "verify-time-single-thread",
            "proof-size",
        ]:
            if value.get(key) is not None:
                bencher_data[common_name].update(value_to_bencher(value[key], key))
    return bencher_data


def main():
    parser = argparse.ArgumentParser(
        description="Run nightly benchmarks and export results"
    )
    parser.add_argument(
        "--export-file",
        required=False,
        type=str,
        help="Export benchmarks results to file (defaults to stdout)",
    )

    args = parser.parse_args()

    benchmarks = run_benchmark_group(EXAMPLES_TO_RUN)

    bencher_data = dict_to_bencher(benchmarks)
    if args.export_file is None:
        print("Couldn't find export file for hashers writing to stdout instead")
        print(json.dumps(bencher_data))
    else:
        with open(args.export_file, "w") as file:
            json.dump(bencher_data, file)


if __name__ == "__main__":
    main()
