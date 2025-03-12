#!/usr/bin/env bash

set -e

export RUST_BACKTRACE=full
CARGO_PROFILE="${CARGO_PROFILE:-test}"
CARGO_EXTRA_FLAGS="${CARGO_EXTRA_FLAGS:-}"
TOOLCHAIN="${CARGO_STABLE:+ +$RUST_VERSION}"
FEATURES="${FEATURES:-}"

# Enable nightly_features unless building for stable
if [ -z "$CARGO_STABLE" ]; then
    FEATURES="$FEATURES --features=nightly_features"
fi

# Do not build examples at this stage by passing "--tests" explicitly
cargo $TOOLCHAIN test --profile $CARGO_PROFILE $FEATURES $CARGO_EXTRA_FLAGS --tests

# First compile all the examples to speed up overall compilation time due to parallelism.
cargo build --profile $CARGO_PROFILE $FEATURES --examples

# Execute examples. 
# Unfortunately there cargo doesn't support executing all examples with a single command.
# Cargo plugins such as "cargo-examples" do support it but without a possibility to specify "release" profile.
for example in examples/*.rs
do
    cargo run --profile $CARGO_PROFILE $FEATURES --example "$(basename "${example%.rs}")" 
done
