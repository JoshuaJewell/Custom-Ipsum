#!/usr/bin/env bash
## Generate placeholder text from a context file
# Usage: ./run.sh [context_file] [max_tokens]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
julia --project="$here" "$here/src/textgen.jl" "$@"
