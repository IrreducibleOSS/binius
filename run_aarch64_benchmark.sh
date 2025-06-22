#!/bin/bash

# Script to build and run binary_zerocheck benchmark on Ubuntu ARM64 with SVE2 optimizations
# Run this script on your Ubuntu ARM64 machine

set -e

echo "Setting up environment for ARM64 SVE2 optimizations..."

# Set SVE2 target features
export RUSTFLAGS="-C target-feature=+sve2,+sve,+neon,+aes"

# Set target CPU to native for maximum optimization
export RUSTFLAGS="$RUSTFLAGS -C target-cpu=native"

echo "Building benchmarks with SVE2 optimizations..."
cargo build --release --benches

echo "Running binary zerocheck benchmark..."
cargo bench --bench binary_zerocheck

echo "Benchmark completed!"
echo ""
echo "To run with different configurations:"
echo "1. With SVE2 disabled: RUSTFLAGS=\"-C target-feature=-sve2\" cargo bench --bench binary_zerocheck"
echo "2. With all SIMD disabled: RUSTFLAGS=\"-C target-feature=-sve2,-sve,-neon\" cargo bench --bench binary_zerocheck"
echo "3. To compare performance differences between configurations" 