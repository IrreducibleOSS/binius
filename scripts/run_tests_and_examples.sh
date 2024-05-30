#!/usr/bin/env bash

set -e

export RUST_BACKTRACE=full

cargo test

# First compile all the examples in release mode to speed up overall compilation time due to parallelism.
cargo build --release --examples

# Execute examples. 
# Unfortunately there cargo doesn't support executing all examples with a single command.
# Cargo plugins such as "cargo-examples" do suport it but without a possibility to specify "release" profile.
for example in examples/*.rs
do
    cargo run --example "$(basename "${example%.rs}")"  --release
done