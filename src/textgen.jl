include("decoder.jl")
include("encoder.jl")
include("tools.jl")

using JLD2, Random
using .Decoder, .Encoder, .Tools

Random.seed!(1234)

# Reading from data
# finewebcontext = read("./data/contexts/fineweb-top5000.txt", String)
# finewebtensors = jldopen("./data/tensordicts/fineweb-top5000.tensordict", "r") do file
#     file["data"]
# end

#print(decode(encode_multiple("./data/contexts/", "localsample", 12)))

context = read("./data/contexts/localsample1.txt", String)
print(decode(encode(context, "sanger"), "sanger"))