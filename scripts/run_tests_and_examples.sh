#!/usr/bin/env bash

export RUST_BACKTRACE=full

cargo test


# Execute examples. 
# Unfortunately there cargo doesn't support executing all examples with a single command.
# Cargo plugins such as "cargo-examples" do suport it but without a possibility to specify "release" profile.
for example in examples/*.rs
do
    cargo run --example "$(basename "${example%.rs}")"  --release
done