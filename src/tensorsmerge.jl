module TensorsMerge

    export merge_tensors

    function merge_inner(d1::Dict{String, Float64}, d2::Dict{String, Float64}, ratio::Float64)
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

    # Function to merge two dictionaries with the given ratio
    function merge_tensors(d1::Dict{String, Dict{String, Float64}}, d2::Dict{String, Dict{String, Float64}}, ratio::Float64)
        merged = deepcopy(d1)

        for (outer_key, inner_dict) in d2
            if haskey(merged, outer_key)
                # Merge inner dictionaries with given ratio
                merged_inner = merge_inner(merged[outer_key], inner_dict, ratio)
                merged[outer_key] = merged_inner
            else
                # Add new outer key from d2
                merged[outer_key] = deepcopy(inner_dict)
            end
        end

        return merged
    end
end