#!/bin/env bash

set -Eeuo pipefail

# Build kernel
echo "Building kernel..."
cargo -Z unstable-options -C kernel build-x86_64

# Perform checks
if [ -e target/x86_64-unknown-none/debug/busyos ]; then
  cargo run --package=xtask --bin check-stack-usage -- target/x86_64-unknown-none/debug/busyos
fi
if [ -e target/x86_64-unknown-none/release/busyos ]; then
  cargo run --package=xtask --bin check-stack-usage -- target/x86_64-unknown-none/release/busyos
fi

# Build loader
echo "Building bootloader..."
cargo -Z unstable-options -C bootloader build-x86_64
