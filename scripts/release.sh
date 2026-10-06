#!/bin/bash
set -euo pipefail

# Release implementation is portable across macOS/Linux with Python 3, Git and Go.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/release.py" "$@"
