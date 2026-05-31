#!/bin/env bash

# Build kernel
cargo build --target x86_64-unknown-none -p busyos

# Build loader
cargo build --target x86_64-unknown-uefi -p busyos_bootloader
