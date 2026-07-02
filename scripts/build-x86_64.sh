#!/bin/env bash

# Build kernel
echo "Building kernel..."
cargo -Z unstable-options -C kernel build-x86_64

# Build loader
echo "Building bootloader..."
cargo -Z unstable-options -C bootloader build-x86_64
