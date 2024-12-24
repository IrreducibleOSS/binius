#!/usr/bin/env bash

set -e

export RUST_BACKTRACE=full
CARGO_PROFILE="${CARGO_PROFILE:-test}"
TOOLCHAIN="${CARGO_STABLE:+ +$RUST_VERSION}"
PACKAGES="${CARGO_STABLE:+ -p binius_utils -p binius_hash -p binius_field -p binius_core}"
FEATURES="${CARGO_STABLE:+ --features=stable_only}"

cargo $TOOLCHAIN test --profile $CARGO_PROFILE $PACKAGES $FEATURES

# Run examples only if CARGO_STABLE is not set since now they won't compile with stable toolchain.
if [ -z "$CARGO_STABLE" ]; then

    # First compile all the examples to speed up overall compilation time due to parallelism.
    cargo build --profile $CARGO_PROFILE --examples

    # Execute examples. 
    # Unfortunately there cargo doesn't support executing all examples with a single command.
    # Cargo plugins such as "cargo-examples" do suport it but without a possibility to specify "release" profile.
    for example in examples/*.rs
    do
        cargo run --profile $CARGO_PROFILE --example "$(basename "${example%.rs}")"
    done
fi