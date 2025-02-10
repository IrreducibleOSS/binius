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
GROESTLP_PERMS = 1 << 14
VISION32B_PERMS = 1 << 14
SHA256_PERMS = 1 << 14
NUM_BINARY_OPS = 1 << 22
NUM_MULS = 1 << 20

HASHER_TO_RUN = {
    r"keccakf": {
        "type": "hasher",
        "display": r"Keccak-f",
        "export": "keccakf-report.csv",
        "args": ["keccakf_circuit", "--", "--n-permutations"],
        "n_ops": KECCAKF_PERMS,
    },
    "groestlp": {
        "type": "hasher",
        "display": r"Grøstl P",
        "export": "groestl-report.csv",
        "args": ["groestl_circuit", "--", "--n-permutations"],
        "n_ops": GROESTLP_PERMS,
    },
    "vision32b": {
        "type": "hasher",
        "display": r"Vision Mark-32",
        "export": "vision32b-report.csv",
        "args": ["vision32b_circuit", "--", "--n-permutations"],
        "n_ops": VISION32B_PERMS,
    },
    "sha256": {
        "type": "hasher",
        "display": "SHA-256",
        "export": "sha256-report.csv",
        "args": ["sha256_circuit", "--", "--n-compressions"],
        "n_ops": SHA256_PERMS,
    },
    "b32_mul": {
        "type": "binary_ops",
        "display": "BinaryField32b mul",
        "export": "b32-mul-report.csv",
        "args": ["b32_mul", "--", "--n-ops"],
        "n_ops": NUM_MULS,
    },
    "u32_add": {
        "type": "binary_ops",
        "display": "u32 add",
        "export": "u32-add-report.csv",
        "args": ["u32_add", "--", "--n-additions"],
        "n_ops": NUM_BINARY_OPS,
    },
    "u32_mul": {
        "type": "binary_ops",
        "display": "u32 mul",
        "export": "u32-mul-report.csv",
        "args": ["u32_mul", "--", "--n-muls"],
        "n_ops": NUM_MULS,
    },
    "xor": {
        "type": "binary_ops",
        "display": "Xor",
        "export": "xor-report.csv",
        "args": ["bitwise_ops", "--", "--op", "xor", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
    "and": {
        "type": "binary_ops",
        "display": "And",
        "export": "and-report.csv",
        "args": ["bitwise_ops", "--", "--op", "and", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
    "or": {
        "type": "binary_ops",
        "display": "Or",
        "export": "or-report.csv",
        "args": ["bitwise_ops", "--", "--op", "or", "--n-u32-ops"],
        "n_ops": NUM_BINARY_OPS,
    },
}

HASHER_BENCHMARKS = {}
BINARY_OPS_BENCHMARKS = {}


def run_benchmark(benchmark_args) -> tuple[bytes, bytes]:
    command = (
        ["cargo", "run", "--release", "--example"]
        + benchmark_args["args"]
        + [f"{benchmark_args['n_ops']}"]
    )
    env_vars_to_run = {
        **os.environ,
        **ENV_VARS,
        "PROFILE_CSV_FILE": benchmark_args["export"],
    }
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
                data.update({"trace_gen_time": int(row[2])})
            elif row[0] == "constraint_system::prove":
                data.update({"proving_time": int(row[2])})
            elif row[0] == "constraint_system::verify":
                data.update({"verification_time": int(row[2])})
    return data


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


def nano_to_milli(nano) -> float:
    return float(nano) / 1000000.0


def nano_to_seconds(nano) -> float:
    return float(nano) / 1000000000.0


def run_and_parse_benchmark(benchmark, benchmark_args) -> tuple[dict, int]:
    data = {}
    stdout = None
    print(f"Running benchmark: {benchmark} with {SAMPLE_SIZE} samples")
    for _ in range(SAMPLE_SIZE):
        stdout, _stderr = run_benchmark(benchmark_args)
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
            data, proof_size = run_and_parse_benchmark(benchmark, benchmark_args)
            benchmark_results[benchmark] = {"proof_size_kib": proof_size}
            data["n_ops"] = benchmark_args["n_ops"]
            data["display"] = benchmark_args["display"]
            data["type"] = benchmark_args["type"]
            benchmark_results[benchmark].update(data)

        except Exception as e:
            print(f"Failed to run benchmark: {benchmark} with error {e} \nExiting...")
            exit(1)
    return benchmark_results


def value_to_bencher(value: Union[list[float], int], throughput: bool = False) -> dict:
    if isinstance(value, list):
        avg_value = sum(value) / len(value)
        max_value = max(value)
        min_value = min(value)
    else:
        avg_value = max_value = min_value = value

    metric_type = "throughput" if throughput else "latency"
    return {
        metric_type: {
            "value": avg_value,
            "upper_value": max_value,
            "lower_value": min_value,
        }
    }


def dict_to_bencher(data: dict) -> dict:
    bencher_data = {}
    for benchmark, value in data.items():
        # Name is of the following format: <benchmark_type>::<benchmark_name>::(trace_gen_time | proving_time | verification_time | proof_size_kib | n_ops)
        common_name = f"{value['type']}::{value['display']}"
        for key in [
            "trace_gen_time",
            "proving_time",
            "verification_time",
            "proof_size_kib",
            "n_ops",
        ]:
            bencher_data[f"{common_name}::{key}"] = value_to_bencher(value[key])
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

    benchmarks = run_benchmark_group(HASHER_TO_RUN)

    bencher_data = dict_to_bencher(benchmarks)
    if args.export_file is None:
        print("Couldn't find export file for hashers writing to stdout instead")
        print(json.dumps(bencher_data))
    else:
        with open(args.export_file, "w") as file:
            json.dump(bencher_data, file)


if __name__ == "__main__":
    main()
