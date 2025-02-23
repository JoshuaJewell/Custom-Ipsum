include("decoder.jl")
include("encoders.jl")
include("tensorsmerge.jl")

using JLD2, Random
using .Decoder, .Encoders, .TensorsMerge

Random.seed!(1234)

finewebcontext = read("../data/contexts/fineweb-top5000.txt", String)
finewebtensors = jldopen("../data/tensordicts/fineweb-top5000.tensordict", "r") do file
    file["data"]
end

# Encode a tensordict from context
function encode(context, mode = "default", end_punctuation = [".", "!", "?"], exclude = [" ", "(", ")", "\"", "*"])
    initT = time()
    if mode == "sanger"
        sanger_encoder(context, fragment_size = 1)
    else
        default_encoder(context, end_punctuation, exclude)
    end
    println("\nEncoded in $(time() - initT) s")
end

stream = false

initT = time()
generated_text = default_decoder(tensors)

#generated_text = decode_beam(tensors)
if !stream
    println("\n",generated_text)
end
println("Decoded in $(time() - initT) s")