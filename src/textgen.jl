include("decoder.jl")
include("encoder.jl")
include("tools.jl")

using Random, Serialization

using .Decoder, .Encoder, .Tools

Random.seed!(123)

# Reading from data
#finewebcontext = read("./data/contexts/fineweb-top5000.txt", String)
 

#print(decode(encode_multiple("./data/contexts/", "localsample", 4), max_tokens = 512))

tensorsmult = encode_multiple(
    "/home/joshua/Documents/paststatuses-split/", #shh, don't tell anyone how badly organised my directory structure is...
    "",
    "",
    26,
    encoder_mode = "sanger",
    fragment_size = 5,
    fragment_groups = 3, 
)

context = read("/home/joshua/Documents/paststatuses.txt", String)
tensorsone = encode(context, "sanger", fragment_size = 5, fragment_groups = 3)

#open("./data/models/wastatustest.ctensors", "w") do file
#    serialize(file, tensors)
#end

#tensors1 = open("./data/models/wastatustest.ctensors", "r") do file
#    deserialize(file)
#end
#tensors2 = open("./data/models/wastatustest.ctensors", "r") do file
#    deserialize(file)
#end

#tensorsmerged = merge_ctensors(tensors1, tensors2)

println("One file.")
println(decode(
    tensorsone,
    temperature=1,
    stream=false,
    show_tokens=false,
    max_tokens=64
))

println("Merged multiple.")
println(decode(
    tensorsmult,
    temperature=1,
    stream=false,
    show_tokens=false,
    max_tokens=64
))

#tensors = encode(context, "sanger", end_punctuation=end_punctuation, exclude=exclude, fragment_size=fragment_size, fragment_groups=fragment_groups),
        
#print(encoder_decoder(context, "sanger", fragment_size = 5, fragment_groups = 3, temperature=0.8, stream=false, show_tokens=false, max_tokens=256))

# discard 'clean' cuts might help preserve coherence?
# use default_encoder to check output of sanger_encoder?
# CompleteTensors tools