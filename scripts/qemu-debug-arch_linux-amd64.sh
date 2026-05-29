#!/bin/env bash

# Check dependencies
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source -- "$SCRIPT_DIR/common.sh"
source -- "$SCRIPT_DIR/dependencies.sh"

# Create esp/efi/boot
TMP_DIR=$(mktemp -d)
ESP_DIR="$TMP_DIR/esp"
EFI_DIR="$TMP_DIR/esp/efi"
BOOT_DIR="$TMP_DIR/esp/efi/boot"
mkdir -p "$BOOT_DIR"

# Automatically clean up
trap 'rm -rf "$TMP_DIR"' EXIT

# Prepare UEFI bootloader
OVMF_PATH="/usr/share/ovmf/x64/"
cp "$OVMF_PATH/OVMF_CODE.4m.fd" "$ESP_DIR/OVMF_CODE.fd"
cp "$OVMF_PATH/OVMF_VARS.4m.fd" "$ESP_DIR/OVMF_VARS.fd"

# Prepare UEFI OS Loader
cp -- "$SCRIPT_DIR/../target/x86_64-unknown-uefi/debug/busyos.efi" "$BOOT_DIR/bootx64.efi"

# Start tmux with QEMU+GDB
tmux new-session -d -s busyos
tmux send-keys -t busyos:0.0 "qemu-system-x86_64 -enable-kvm \
    -drive if=pflash,format=raw,readonly=on,file=\"$ESP_DIR/OVMF_CODE.fd\" \
    -drive if=pflash,format=raw,readonly=on,file=\"$ESP_DIR/OVMF_VARS.fd\" \
    -drive format=raw,file=fat:rw:\"$ESP_DIR\" \
    -s -S \
    $qemu_FLAGS" Enter
tmux split-window -h -t busyos:0
tmux send-keys -t busyos:0.1 "gdb -ex 'target remote :1234'" Enter
tmux attach-session -t busyos
