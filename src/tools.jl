module Tools
    include("types.jl")
    include("encoder.jl")
    include("decoder.jl")

    using .Types, .Encoder, .Decoder

    export encode_multiple, encoder_decoder, encode_incremental

    function encoder_decoder(
        context,
        mode = "default";
        end_punctuation = [".", "!", "?"], 
        exclude = [" ", "(", ")", "\"", "*"], 
        fragment_size = 1,
        fragment_groups = 1,
        max_tokens = 128,
        stream = false,
        stream_rate = 1000,
        show_tokens = false,
        temperature = 1
    )
        return decode(
            encode(context, mode, end_punctuation=end_punctuation, exclude=exclude, fragment_size=fragment_size, fragment_groups=fragment_groups),
            max_tokens=max_tokens,
            stream=stream,
            stream_rate=stream_rate,
            show_tokens=show_tokens,
            temperature=temperature
    )
    end

    """
        function encode_multiple(path_to_context = "../data/contexts/", context_filename = "context", context_file_no; mode = "equal")
    
    Encodes multiple contexts from a set of files and merges the weights. Store contexts in format:
        context1.txt
        context2.txt
        etc...

    ## Arguments
    - `path_to_context` (default: "../data/contexts/"): Where your context files to encode and merge are.
    - `context_filename` (default: "context"): The unchanged part of your contexts' filenames.
    - `context_file_no` (default: 2): The number of context files, indexed from 1.

    ## Keyword Arguments
    - `merge_mult` (optional, default: 1): How much weight to give added tensordict. (WIP)
    - `encoder_mode` (optional, default: "default"): The encoder mode. Can be "default", or "sanger".
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sanger".
    - `fragment_groups` (optional, default: 1): How many different fragment sizes should be parsed (high values not recommended). Only relevant if `mode` is "sanger" and `fragment_size` is specified.

    """
    function encode_multiple(
        path_to_context = "../data/contexts/",
        context_filename = "context",
        context_file_no = 2;
        merge_mult = 1,
        encoder_mode = "sanger",
        exclude = [""],
        fragment_size = 1,
        fragment_groups = 1
    )
        encoder_mode = lowercase(encoder_mode)

        contexts = []
        for i in 1:context_file_no
            context = read("$path_to_context$context_filename$i.txt", String)
            push!(contexts, context)
        end

        println()
        merged_tensors = nothing
        contextcount = length(contexts)
        args = ""

        for (idx, context) in enumerate(contexts)
            print("\x1b[1A\rProcessing file $idx of $contextcount\n...")
            if encoder_mode == "sanger"
                tensor = sanger_encoder(context, fragment_size, fragment_groups, exclude, false)
                args = "Fragmentation: $fragment_size by $fragment_groups. $exclude excluded."
            else
                tensor = default_encoder(context, verbose=false)
                args = "Sentence enders: $end_punctuation; Preserved tokens: $preserve_tokens. $exclude excluded."
            end

            if merged_tensors === nothing
                merged_tensors = tensor
            else
                merged_tensors = merge_tensordicts(merged_tensors, tensor)
            end
        end

        tensors = CompleteTensors(
            Header(encoder_mode, args),
            merged_tensors,
            Dict{String, Dict{String, Float64}}(), 
            [""]
        )

        return tensors
    end

    """
        function merge_tensordicts(d1::Dict{String, Dict{String, Float64}}, d2::Dict{String, Dict{String, Float64}}; ratio::Float64 = 0.5)
    
    Merges the weights from tensordicts.

    ## Arguments
    - `d1`: Tensordict the first.
    - `d2`: Tensordict the second.
    """
    function merge_tensordicts(
        d1::Dict{String, Dict{String, Float64}},
        d2::Dict{String, Dict{String, Float64}};
        merge_mult = 1
    )
        for (outer_key, inner_dict) in d2
            if haskey(d1, outer_key)
                merge!(d1[outer_key], inner_dict)
            else
                d1[outer_key] = inner_dict
            end
        end

        return d1
    end

    function merge_inner_merge_tensordicts(
        d1::Dict{String, Float64},
        d2::Dict{String, Float64};
        merge_mult = 1
    )
        for (key, value) in d2
            if haskey(d1, key)
                d1[key] += value
            else
                d1[key] = value
            end
        end

        return d1
    end
end