include("decoder.jl")
include("encoder.jl")
include("tensorsmerge.jl")

using JLD2, Random
using .Decoder, .Encoder, .TensorsMerge

Random.seed!(1234)

finewebcontext = read("../data/contexts/fineweb-top5000.txt", String)
finewebtensors = jldopen("../data/tensordicts/fineweb-top5000.tensordict", "r") do file
    file["data"]
end

stream = false

initT = time()
generated_text = default_decoder(tensors)

#generated_text = decode_beam(tensors)
if !stream
    println("\n",generated_text)
end
println("Decoded in $(time() - initT) s")