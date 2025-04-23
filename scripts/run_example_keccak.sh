#!/bin/bash

# Ensure the script exits on the first failure
set -e

# ---------------------------------------
# Variables and Constants
# ---------------------------------------

# Repository root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
cd "${ROOT_DIR}"

# Scripts URLs
OPEN_TRACE_UI="./scripts/open_trace_in_ui"
TRACE_PROCESSOR="./scripts/trace_processor"
OPEN_TRACE_UI_URL="https://github.com/google/perfetto/raw/main/tools/open_trace_in_ui"
TRACE_PROCESSOR_URL="https://get.perfetto.dev/trace_processor"

# Trace output directory
TRACE_DIR="perfetto-traces"
mkdir -p "${TRACE_DIR}"

# Timestamp and Git information
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9._-]/-/g')
COMMIT_HASH=$(git rev-parse --short HEAD)
ARCHITECTURE=$(uname -m)
HOSTNAME=$(hostname)

# ---------------------------------------
# Dependency Checks and Setup
# ---------------------------------------

# Download 'open_trace_in_ui' if not present
if [ ! -f "${OPEN_TRACE_UI}" ]; then
    echo "Downloading open_trace_in_ui..."
    curl -fSL "${OPEN_TRACE_UI_URL}" -o "${OPEN_TRACE_UI}"
    chmod +x "${OPEN_TRACE_UI}"
fi

# Download 'trace_processor' if not present
if [ ! -f "${TRACE_PROCESSOR}" ]; then
    echo "Downloading trace_processor..."
    curl -fSL "${TRACE_PROCESSOR_URL}" -o "${TRACE_PROCESSOR}"
    chmod +x "${TRACE_PROCESSOR}"
fi

# ---------------------------------------
# Determine Git Repo Status
# ---------------------------------------

# Check for uncommitted changes
dirty_flag=""
if [ -n "$(git status --porcelain)" ]; then
    dirty_flag="-dirty"
    diff_file="${TRACE_DIR}/${TIMESTAMP}-${BRANCH_NAME}-${COMMIT_HASH}${dirty_flag}-${ARCHITECTURE}-${HOSTNAME}.diff"
    echo "Uncommitted changes detected, saving diff to ${diff_file}"
    git diff > "${diff_file}"
fi

# Perfetto trace file path
export PERFETTO_TRACE_FILE_PATH="${TRACE_DIR}/${TIMESTAMP}-${BRANCH_NAME}-${COMMIT_HASH}${dirty_flag}-${ARCHITECTURE}-${HOSTNAME}.perfetto-trace"

# ---------------------------------------
# Execution Configuration
# ---------------------------------------

# Control Rayon threading:
# Multi-threaded: leave unset or set RAYON_NUM_THREADS=0
# Single-threaded: export RAYON_NUM_THREADS=1
# uncomment below for single-threaded execution
# export RAYON_NUM_THREADS=1

export RUSTFLAGS="-C target-cpu=native"

# ---------------------------------------
# Run Rust Example with Perfetto Tracing
# ---------------------------------------

echo "Running Rust example with Perfetto tracing enabled..."
cargo run --release --features perfetto --example keccakf_circuit -- \
    --n-permutations $((2 ** 13)) \
    --repeat 10 \
    --warm-up

# Short wait to ensure trace data is finalized
sleep 1

# ---------------------------------------
# Process and Display Trace Data
# ---------------------------------------

# Analyze trace data using trace_processor
"${TRACE_PROCESSOR}" -Q "
WITH span_summary AS (
  SELECT
    name,
    AVG(dur / 1e6) AS avg_duration_ms,
    GROUP_CONCAT(dur / 1e6) AS durations_ms
  FROM slice
  WHERE name LIKE '[phase]%'
  GROUP BY name
)

SELECT name, avg_duration_ms, durations_ms FROM span_summary

UNION ALL

SELECT '[phase] TOTAL',
       SUM(avg_duration_ms) AS avg_duration_ms,
       NULL AS durations_ms
FROM span_summary;
" "${PERFETTO_TRACE_FILE_PATH}"

# Open Perfetto trace in UI without automatically opening the browser
"${OPEN_TRACE_UI}" --no-open-browser "${PERFETTO_TRACE_FILE_PATH}"
