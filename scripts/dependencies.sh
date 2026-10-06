#!/bin/env bash

set -Eeuo pipefail

# Directory of current script
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source -- "$SCRIPT_DIR/common.sh"

if is_distro "$DISTRO_ARCH_LINUX"; then
  source -- "$SCRIPT_DIR/dependencies-arch_linux-x86_64.sh"
else
  err "Unknown distribution for installer: $ID"
  exit $EXIT_CODE_FAILURE
fi
