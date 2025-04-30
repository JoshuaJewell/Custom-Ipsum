module Tools
    include("encoder.jl")
    include("decoder.jl")

    using .Encoder, .Decoder

    export encode_multiple, merge_tensors, encoder_decoder

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
        temperature = 0
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
    - `merge_mode` (optional, default: "equal"): The merging mode. Can be "equal", or "weighted". (WIP)
    - `encoder_mode` (optional, default: "default"): The encoder mode. Can be "default", or "sanger".

    """
    function encode_multiple(
        path_to_context = "../data/contexts/",
        context_filename = "context",
        context_file_no = 2;
        merge_mode = "equal",
        encoder_mode = "default"
    )
        merge_mode = lowercase(merge_mode)
        encoder_mode = lowercase(encoder_mode)

        contexts = []
        for i in 1:context_file_no
            context = read("$path_to_context$context_filename$i.txt", String)
            push!(contexts, context)
        end

        tensors = [encode(context, encoder_mode) for context in contexts]

        merged_tensors = tensors[1]

        for i in eachindex(tensors[2:end])
            ratio = 1.0 / i
            merged_tensors = merge_tensors(merged_tensors, tensors[i + 1], "weighted", ratio = ratio)
        end

        return merged_tensors
    end

    """
        function merge_tensors(d1::Dict{String, Dict{String, Float64}}, d2::Dict{String, Dict{String, Float64}}; ratio::Float64 = 0.5)
    
    Merges the weights from tensordicts.

    ## Arguments
    - `d1`: Tensordict the first.
    - `d2`: Tensordict the second.

    ## Keyword Arguments
    - `ratio` (optional, default: 0.5): The ratio by which. Only relevant if `mode` is "ratio".
    - `mode` (optional, default: "weighted"): The merging mode. Can be "ratio", or "weighted". (WIP)
    """
    function merge_tensors(
        d1::Dict{String, Dict{String, Float64}},
        d2::Dict{String, Dict{String, Float64}},
        mode = "weighted";
        ratio::Float64 = 0.5
    )
        mode = lowercase(mode)

        merged_tensors = deepcopy(d1)

        for (outer_key, inner_dict) in d2
            if haskey(merged_tensors, outer_key)
                # Merge inner dictionaries with given ratio
                merged_inner = merge_inner_merge_tensors(merged_tensors[outer_key], inner_dict, ratio)
                merged_tensors[outer_key] = merged_inner
            else
                # Add new outer key from d2
                merged_tensors[outer_key] = deepcopy(inner_dict)
            end
        end

        return merged_tensors
    end

    function merge_inner_merge_tensors(
        d1::Dict{String, Float64},
        d2::Dict{String, Float64},
        ratio::Float64
    )
        merged_inner = Dict{String, Float64}()

        # Get all unique keys from both dictionaries
        all_keys = union(keys(d1), keys(d2))

        for key in all_keys
            val1 = get(d1, key, 0.0)
            val2 = get(d2, key, 0.0)

            # Compute the weighted sum
            new_val = val1 * (1 - ratio) + val2 * ratio

            if new_val != 0.0
                merged_inner[key] = new_val
            end
        end

        return merged_inner
    end
end