using CustomIpsum
using Random, Serialization

Random.seed!(123)

const REPO = joinpath(@__DIR__, "..")

## Read command-line arguments
# Usage: julia src/textgen.jl [context_file] [max_tokens]
# Both are optional; a relative context path resolves against the working directory.
context_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "data", "contexts", "macbeth.txt")
max_tokens = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 512

## Build a model from the chosen context and generate from it
context = read(context_path, String)
tensors = encode(context, "sanger", fragment_size = 5, fragment_groups = 3)

## Persist the model to a local-scoped (gitignored) output
# The name follows the context so different corpora do not overwrite one another.
context_name = splitext(basename(context_path))[1]
open(joinpath(REPO, "data", "models", "local-$(context_name).ctensors"), "w") do file
    serialize(file, tensors)
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
