using CustomIpsum
using Random

Random.seed!(123)

const REPO = joinpath(@__DIR__, "..")

## Read command-line arguments
# Usage: julia src/textgen.jl [context_or_model] [max_tokens]
# The first argument may be a text context (any extension) or a saved ".ctensors"
# model; both are optional. A relative path resolves against the working directory.
input_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "data", "contexts", "macbeth.txt")
max_tokens = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 512

## Obtain a model: load a saved one, or encode a text context and save it
if endswith(lowercase(input_path), ".ctensors")
    tensors = load_model(input_path)
else
    context = read(input_path, String)
    tensors = encode(context, "sanger", fragment_size = 5, fragment_groups = 3)
    context_name = splitext(basename(input_path))[1]
    save_model(joinpath(REPO, "data", "models", "local-$(context_name).ctensors"), tensors)
end

println(decode(
    tensors,
    temperature = 1,
    stream = false,
    show_tokens = false,
    max_tokens = max_tokens
))

## Design notes for later
# Independent chains with their own "personalities" to create dialogue. This
# will require some global coherence to make any sense at all.
