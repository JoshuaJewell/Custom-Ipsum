include("decoder.jl")
include("encoder.jl")
include("tools.jl")

using Random, Serialization

using .Decoder, .Encoder, .Tools

Random.seed!(123)

# Reading from data
#finewebcontext = read("./data/contexts/fineweb-top5000.txt", String)
 

#print(decode(encode_multiple("./data/contexts/", "localsample", 4), max_tokens = 512))


context = read("./data/contexts/localcontext.txt", String)

#tensors = encode(context, "sanger", fragment_groups = 6)

#open("./data/tensordicts/local123.tensordict", "w") do file
#    serialize(file, tensors)
#end

#tensors = open("./data/tensordicts/localtensors.tensordict", "r") do file
#    deserialize(file)
#end

#tensors = encode(context, "sanger", end_punctuation=end_punctuation, exclude=exclude, fragment_size=fragment_size, fragment_groups=fragment_groups),
        
print(encoder_decoder(context, "sanger", fragment_size = 5, fragment_groups = 3, temperature=0.8, stream=false, show_tokens=false, max_tokens=256))

# discard 'clean' cuts might help preserve coherence?