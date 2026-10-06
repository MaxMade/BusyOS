#!/bin/env bash

set -Eeuo pipefail

# Default for QEMU_FLAGS
QEMU_FLAGS="${QEMU_FLAGS:-}"

# Default for CPU_NUM, the number of cores QEMU emulates
CPU_NUM="${CPU_NUM:-4}"

# Check dependencies
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source -- "$SCRIPT_DIR/common.sh"
source -- "$SCRIPT_DIR/dependencies.sh"

# Create esp/efi/boot
TMP_DIR=$(mktemp -d)
ESP_DIR="$TMP_DIR/esp"
BOOT_DIR="$TMP_DIR/esp/efi/boot"
mkdir -p "$BOOT_DIR"

# Automatically clean up
trap 'rm -rf "$TMP_DIR"' EXIT

# Prepare UEFI bootloader
OVMF_PATH="/usr/share/ovmf/x64/"
cp "$OVMF_PATH/OVMF_CODE.4m.fd" "$ESP_DIR/OVMF_CODE.fd"
cp "$OVMF_PATH/OVMF_VARS.4m.fd" "$ESP_DIR/OVMF_VARS.fd"

# Prepare UEFI OS Loader
cp -- "$SCRIPT_DIR/../target/x86_64-unknown-uefi/debug/busyos-bootloader.efi" "$BOOT_DIR/bootx64.efi"

# Prepare BUSYOS
cp -- "$SCRIPT_DIR/../target/x86_64-unknown-none/debug/busyos" "$BOOT_DIR/busyos.elf"

# Start tmux with QEMU+GDB
tmux new-session -d -s busyos
tmux send-keys -t busyos:0.0 "qemu-system-x86_64 -cpu max,+pdpe1gb -smp $CPU_NUM -m 2G \
    -drive if=pflash,format=raw,readonly=on,file=\"$ESP_DIR/OVMF_CODE.fd\" \
    -drive if=pflash,format=raw,readonly=on,file=\"$ESP_DIR/OVMF_VARS.fd\" \
    -drive format=raw,file=fat:rw:\"$ESP_DIR\" \
    -d int,cpu_reset,guest_errors -D qemu.log \
    -s -S $QEMU_FLAGS" Enter
tmux split-window -h -t busyos:0
tmux send-keys -t busyos:0.1 "gdb -ex 'target remote :1234'" Enter
tmux attach-session -t busyos
