#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# This connects GDB with X11, if the container is started with X11 forwarding.
if [ -n "${DISPLAY}" ]; then
  echo "set environment DISPLAY ${DISPLAY}" >> "/root/.gdbinit"
fi

exec "$@"
