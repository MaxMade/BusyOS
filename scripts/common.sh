#!/bin/env bash
source /etc/os-release

EXIT_CODE_SUCCESS=0
EXIT_CODE_FAILURE=1
EXIT_CODE_MISSING_PACKAGE=2

DISTRO_ARCH_LINUX="arch"
DISTRO_DEBIAN="debian"
DISTRO_UBUNTU="ubuntu"
DISTRO_FEDORA="fedora"
DISTRO_RHEL="rhel"

# Usage: die_missing_package "extra/qemu-full"
function die_missing_package() {
  if is_distro "$DISTRO_ARCH_LINUX"; then
    err "Package \"$1\" missing. Please use \"pacman -S $1\" to install it."
  elif is_distro "$DISTRO_DEBIAN" "$DISTRO_UBUNTU"; then
    err "Package \"$1\" missing. Please use \"apt-get install $1\" to install it."
  elif is_distro "$DISTRO_FEDORA" "$DISTRO_RHEL"; then
    err "Package \"$1\" missing. Please use \"dnf install $1\" to install it."
  else
    err "Package \"$1\" missing."
  fi
  exit $EXIT_CODE_MISSING_PACKAGE
}

# Usage: err "Invalid command"
function err() {
  printf -- "$@\n" >&2
}

# Usage: is_distro $DISTRO_ARCH_LINUX
is_distro() {
  for d in "$@"; do
    [[ "$ID" == "$d" || "$ID_LIKE" == *"$d"* ]] && return 0
  done
  return 1
}
