module
    include("encoder.jl")

    using .Encoder

    export encode_multiple

    """
        function encode_multiple(path_to_context = "./", context_filename = "sample", context_file_no = 12, output_tensordict_to)
    
    Encodes multiple contexts from a set of files and merges the weights. Store contexts in format:
        context1.txt
        context2.txt
        etc...

    ## Arguments
    - `path_to_context` (default: "./"): Where your context files to encode and merge are.
    - `context_filename` (default: "context"): The unchanged part of your contexts' filenames.
    - `context_file_no`: The number of context files, indexed from 1.

    ## Keyword Arguments
    - `mode` (optional, default: "equal"): The merging mode. Can be "equal", or "weighted". (WIP)
    """
    function encode_multiple(path_to_context = "./", context_filename = "context", context_file_no; mode = "equal")
        mode = lowercase(mode)

        contexts = []
        for i in 1:context_file_no
            context = read("$path_to_context$context_filename$i.txt", String)
            push!(contexts, context)
        end

        tensors = [encode(context) for context in contexts]

        merged_tensors = tensors[1]

        for i in eachindex(tensors[2:end])
            ratio = 1.0 / i
            merged_tensors = merge_tensors(merged_tensors, tensors[i + 1], ratio)
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
    function merge_tensors(d1::Dict{String, Dict{String, Float64}}, d2::Dict{String, Dict{String, Float64}}; mode = "weighted", ratio::Float64 = 0.5)
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

    function merge_inner_merge_tensors(d1::Dict{String, Float64}, d2::Dict{String, Float64}, ratio::Float64)
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