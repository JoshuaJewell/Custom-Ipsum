include("decoder.jl")
include("encoder.jl")
include("tools.jl")

using JLD2, Random
using .Decoder, .Encoder, .Tools

Random.seed!(12345)

# Reading from data
# finewebcontext = read("./data/contexts/fineweb-top5000.txt", String)
# finewebtensors = jldopen("./data/tensordicts/fineweb-top5000.tensordict", "r") do file
#     file["data"]
# end

#print(decode(encode_multiple("./data/contexts/", "localsample", 4), max_tokens = 512))


context = read("./data/contexts/localsample1.txt", String)
dict = encode(context, "sanger")
print(decode(dict, "sanger", stream=true))