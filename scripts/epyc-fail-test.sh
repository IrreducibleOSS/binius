#!/usr/bin/env bash

# This test case will fail on EPYC AWS instances due to improper inlining when
# compiling with target-cpu=native.  See epyc-fail.env and epyc-fail-lld.env
# for test setup.

set -e
set -x

export RUST_BACKTRACE=0
CARGO_PROFILE="${CARGO_PROFILE:-test}"
CARGO_EXTRA_FLAGS="${CARGO_EXTRA_FLAGS:-}"
FEATURES="${FEATURES:-}"
# Enable nightly_features unless building for stable
if [ ! -z "$CARGO_STABLE" ]; then
    FEATURES="$FEATURES --no-default-features --features=rayon"
fi
CARGO="cargo ${CARGO_STABLE:+ +$RUST_VERSION}"

# Do not build examples at this stage by passing "--tests" explicitly
$CARGO test -p binius_circuits --lib  --profile $CARGO_PROFILE $FEATURES $CARGO_EXTRA_FLAGS -- arithmetic::mul::tests::test_mul --exact --show-output

