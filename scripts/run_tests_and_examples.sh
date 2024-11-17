#!/usr/bin/env bash

set -e

export RUST_BACKTRACE=full
CARGO_PROFILE="${CARGO_PROFILE:-test}"

cargo test --profile $CARGO_PROFILE

# First compile all the examples to speed up overall compilation time due to parallelism.
cargo build --profile $CARGO_PROFILE --examples

# Execute examples. 
# Unfortunately there cargo doesn't support executing all examples with a single command.
# Cargo plugins such as "cargo-examples" do suport it but without a possibility to specify "release" profile.
for example in examples/*.rs
do
    cargo run --profile $CARGO_PROFILE --example "$(basename "${example%.rs}")"
done