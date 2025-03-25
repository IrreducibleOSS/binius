#!/usr/bin/env bash

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

