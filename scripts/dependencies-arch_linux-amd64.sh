#!/bin/env bash

# Directory of current script
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source -- "$SCRIPT_DIR/common.sh"

# Check if OVMF exists
OVMF_PATH="/usr/share/ovmf/x64/"
if [ ! -f "$OVMF_PATH/OVMF_CODE.4m.fd" ] || [ ! -f "$OVMF_PATH/OVMF_VARS.4m.fd" ]; then
  die_missing_package "extra/edk2-ovmf"
fi

# Check if QEMU exists
QEMU=/usr/bin/qemu-system-x86_64
if [ ! -f "$QEMU" ]; then
  die_missing_package "extra/qemu-full"
fi

# Check if tmux exists
QEMU=/usr/bin/tmux
if [ ! -f "$QEMU" ]; then
  die_missing_package "extra/tmux"
fi
