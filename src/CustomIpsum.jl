## Custom Ipsum: a Markov-chain placeholder-text generator
module CustomIpsum

include("types.jl")
include("utils.jl")
include("encoder.jl")
include("decoder.jl")
include("tools.jl")

using .Encoder, .Decoder, .Tools

export encode, decode, encode_multiple, encoder_decoder, merge_ctensors

end
