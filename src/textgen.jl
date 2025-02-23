include("decoders.jl")
include("encoders.jl")
include("tensorsmerge.jl")

using JLD2, Random
using .Decoders, .Encoders, .TensorsMerge

Random.seed!(1234)

#initT = time()
#con = read("E:/CommonCorpus/d/cc-sample_top100_sorted.txt", String)
#con = encode(con)
#println("\nEncoded in $(time() - initT) s")
#

tensors = jldopen("tensordicts/samplemerged@7E-4.tensordict", "r") do file
    file["data"]
end
#tensors2 = jldopen("tensordicts/fineweb-top5000.tensordict", "r") do file
#    file["data"]
#end
#tensors = merge_tensors(tensors1, tensors2, 0.0007)
#
##tensors = jldopen("sample.tensordict", "r") do file
##    file["data"]
##end
#jldopen("tensordicts/samplemerged@7E-4.tensordict", "w") do file
#    file["data"] = tensors
#end

#context = read("contexts/localsample3.txt", String)
#tensors = sanger_encoder(context)

stream = false

initT = time()
generated_text = default_decoder(tensors)

#generated_text = decode_beam(tensors)
if !stream
    println("\n",generated_text)
end
println("Decoded in $(time() - initT) s")