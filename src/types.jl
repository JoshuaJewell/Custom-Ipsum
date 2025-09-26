module Types

    export Header, CompleteTensors, Vocabulary

    ## CompleteTensors V1
    #struct Header
    #    encoding_method::String
    #    metadata::String
    #end

    #struct Metadata
    #    tensor_type::Int
    #    tensor_version::Int
    #    kwargs::Kwargs
    #    comments::String
    #end

    #struct Kwargs
    #    end_punctuation::Vector{String}
    #    exclude::Vector{String}
    #    fragment_size::Int
    #    fragment_groups::Int
    #end

    #struct CompleteTensors
    #    header::Header
    #    forward_markov::Dict{String, Dict{String, Float64}}
    #    reverse_markov::Dict{String, Dict{String, Float64}}
    #    token_index::Vector{String}
    #end

    #struct BasicTensors
    #    header::Header
    #    forward_markov::Dict{String, Dict{String, Float64}}
    #end

    struct Vocabulary
        token_to_id::Dict{String, Int}
        id_to_token::Vector{String}
    end

    ## CompleteTensors V0
    struct Header
        encoding_method::String
        metadata::String
    end
    
    #struct CompleteTensors
    #    header::Header
    #    forward_markov::Dict{String, Dict{String, Float64}}
    #    reverse_markov::Dict{String, Dict{String, Float64}}
    #    token_index::Vector{String}
    #end

    struct CompleteTensors
        header::Header
        vocabulary::Vocabulary
    #   forward_markov::Dict{String, Dict{String, Float64}}
        forward_markov::Dict{Int, Dict{Int, Float64}}
    end
end