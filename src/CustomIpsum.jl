## Custom Ipsum: a Markov-chain placeholder-text generator
module CustomIpsum

include("types.jl")
include("utils.jl")
include("modelio.jl")
include("encoder.jl")
include("decoder.jl")
include("tools.jl")
include("dialogue.jl")

using .Encoder, .Decoder, .Tools, .ModelIO, .Dialogue

export encode, decode, encode_multiple, encoder_decoder, merge_ctensors, save_model, load_model, dialogue, beginnings, endings

end
