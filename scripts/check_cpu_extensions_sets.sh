#!/usr/bin/env bash

# This script check that we do not use CPU extensions that are not supported by the target platform.
# We are using the Rust intrinsics pretty heavily and it's not a compile-time error if you have an intrinsic
# which produces a CPU instruction that is not supported according to `target_features` set.
# This script builds the libraries for different CPU architectures and uses a tool that extracts
# the assembly instructions and reports which instruction set extensions are used.

set -ex

# Download the instruction checking tool
mkdir -p target/instruction-set-analyzer
cd target/instruction-set-analyzer
REPO="IrreducibleOSS/instruction-set-analyzer"
ASSET_NAME_PATTERN="instruction-set-analyzer"
API_URL="https://api.github.com/repos/$REPO/releases/latest"
ASSET_URL=$(curl -s $API_URL | jq -r ".assets[] | select(.name | test(\"$ASSET_NAME_PATTERN\")) | .browser_download_url")
curl -L -o instruction-set-analyzer $ASSET_URL
chmod a+x instruction-set-analyzer
cd ../..
INSTRUCTIONS_SET_ANALYZER="./target/instruction-set-analyzer/instruction-set-analyzer"

# Function to check the libraries for specific CPU extensions
# We check libraries but not binaries, as binaries contain code from the 3rd party dependencies
# which can detect the CPU extensions at runtime, so there may be assembly instructions
# that are not executed at runtime (at least `rand` crate does this).
check_rlibs() {
  local cpu_arch="$1"
  local bad_instr_regex="$2"
  local bad_instr_name="$3"
  local target_dir="target/$cpu_arch"

  # build the libraries for the specified CPU architecture
  mkdir -p "$target_dir"
  RUSTFLAGS="-C target-cpu=$cpu_arch" cargo build --target-dir "$target_dir" --lib \
    --message-format=json \
    | jq -r '
      select(.reason == "compiler-artifact")
      | .filenames[] | select(endswith(".rlib"))' \
    | grep -v "/deps" \
     > "$target_dir/rlib-paths.txt"

  # check that each of the libraries does not contain the wrong instructions
  while IFS= read -r rlib_path; do
    echo "Checking $rlib_path"
    $INSTRUCTIONS_SET_ANALYZER "$rlib_path" | grep -Ei "$bad_instr_regex" > "$target_dir/wrong_instructions.txt" 2>/dev/null || true
    if [ -s "$target_dir/wrong_instructions.txt" ]; then
      echo "Error: $bad_instr_name instructions found in $rlib_path:" >&2
      cat "$target_dir/wrong_instructions.txt" >&2
      exit 1
    fi
  done < "$target_dir/rlib-paths.txt"
}

# platform with SSE, but no AVX or GFNI
check_rlibs "core_2_duo_sse4_1" "avx|gfni" "AVX or GFNI"
# platform with AVX2, but no GFNI
check_rlibs "haswell" "avx512|gfni" "AVX512 or GFNI"
# platform with AVX2 and GFNI, but no AVX512
check_rlibs "znver2" "avx512" "AVX512"

echo "All checks passed successfully."
