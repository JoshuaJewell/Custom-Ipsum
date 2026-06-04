#!/usr/bin/env bash
## Generate placeholder text from a context file or a saved model
# Usage: ./run.sh [context_or_model] [max_tokens]
# The first argument may be a text context or a saved .ctensors model.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
julia --project="$here" "$here/src/textgen.jl" "$@"
