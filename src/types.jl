module Types

    export Header, Metadata, Kwargs, CompleteTensors

    ## CompleteTensors V1
    #struct Header
    #    encoding_method::String
    #    metadata::String
    #end

    #struct Metadata
    #    ctensor_version::Int
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

    ## CompleteTensors V0
    struct Header
        encoding_method::String
        metadata::String
    end
    
    struct CompleteTensors
        header::Header
        forward_markov::Dict{String, Dict{String, Float64}}
        reverse_markov::Dict{String, Dict{String, Float64}}
        token_index::Vector{String}
    end
end