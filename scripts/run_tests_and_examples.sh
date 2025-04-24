#!/usr/bin/env bash
set -e
set -x

export RUST_BACKTRACE=full
CARGO_PROFILE="${CARGO_PROFILE:-test}"
CARGO_EXTRA_FLAGS="${CARGO_EXTRA_FLAGS:-}"
FEATURES="${FEATURES:-}"
# Enable nightly_features unless building for stable
if [ ! -z "$CARGO_STABLE" ]; then
    FEATURES="$FEATURES --no-default-features --features=rayon"
fi
CARGO="cargo ${CARGO_STABLE:+ +$RUST_VERSION} -Z timings"
export CARGO_LOG=debug

env

# Do not build examples at this stage by passing "--tests" explicitly
$CARGO test --profile $CARGO_PROFILE $FEATURES $CARGO_EXTRA_FLAGS --tests

# First compile all the examples to speed up overall compilation time due to parallelism.
$CARGO build --profile $CARGO_PROFILE $FEATURES $CARGO_EXTRA_FLAGS --examples

# Execute examples. 
# Unfortunately there cargo doesn't support executing all examples with a single command.
# Cargo plugins such as "cargo-examples" do support it but without a possibility to specify "release" profile.
for example in examples/*.rs
do
    $CARGO run --profile $CARGO_PROFILE $FEATURES $CARGO_EXTRA_FLAGS --example "$(basename "${example%.rs}")" 
done
