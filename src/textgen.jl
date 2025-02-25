include("decoder.jl")
include("encoder.jl")
include("tools.jl")

using Random, Serialization

using .Decoder, .Encoder, .Tools

Random.seed!(123)

# Reading from data
# finewebcontext = read("./data/contexts/fineweb-top5000.txt", String)
 

#print(decode(encode_multiple("./data/contexts/", "localsample", 4), max_tokens = 512))


context = read("./data/contexts/localreddwarf.txt", String)

tensors = encode(context, "sanger", fragment_groups = 2)

open("./data/tensordicts/macbeth.tensordict", "w") do file
    serialize(file, tensors)
end

tensors = open("./data/tensordicts/macbeth.tensordict", "r") do file
    deserialize(file)
end

print(decode(tensors, max_tokens=1024))