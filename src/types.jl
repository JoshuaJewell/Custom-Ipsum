module Types

    export Header, CompleteTensors

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