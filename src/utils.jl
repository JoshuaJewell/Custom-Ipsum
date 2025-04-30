module Utils

    export average_word_length, sanger_split, recapitalise!, normalize_weights, default_sampler

    function average_word_length(
        text::String,
        smallest_word=3
    )
        # Split the text into words
        words = split(text)
        filter!(word -> length(word) >= smallest_word, words)

        # Count total words and total letters
        total_words = length(words)
        total_letters = sum(length(word) for word in words)

        return total_letters / total_words
    end

    function sanger_split(context, fragment_size, fragment_groups = 1)
        result = []
        
        # Do regular split (creates associations between fragments of the same group)
        for group in 1:fragment_groups
            push!(result, "\n")
            group_array = sanger_split_base(context, fragment_size)
            append!(result, group_array)
            fragment_size += 1
        end
    
        # Do alternating split (creates associations between fragments of different groups)
        sizes = fragment_size:(fragment_size + fragment_groups - 1)
        if fragment_groups > 1
            for size1 in sizes
                for size2 in sizes
                    if size1 != size2
                        alt_array = sanger_split_alt(context, size1, size2)
                        append!(result, alt_array)
                    end
                end
            end
        end
    
        return result
    end
    
    function sanger_split_base(context, fragment_size)
        n = ncodeunits(context)
        result = Vector{String}()
    
        for offset in 1:fragment_size
            current_pos = offset
            while current_pos <= n
                end_pos = current_pos
                for _ in 1:(fragment_size - 1)
                    if end_pos > n
                        break
                    end
                    end_pos = nextind(context, end_pos)
                end
    
                token = context[current_pos:prevind(context, end_pos)]
                push!(result, token)
                current_pos = nextind(context, end_pos - 1)
            end
        end
    
        return result
    end
    
    function sanger_split_alt(context, size1, size2)
        n = ncodeunits(context)
        result = Vector{String}()
    
        for offset in 1:min(size1, size2)
            current_pos = offset
            current_size_idx = 1
            sizes = [size1, size2]
    
            while current_pos <= n
                current_size = sizes[current_size_idx]
                end_pos = current_pos
                for _ in 1:(current_size - 1)
                    if end_pos > n
                        break
                    end
                    end_pos = nextind(context, end_pos)
                end
    
                token = context[current_pos:prevind(context, end_pos)]
                push!(result, token)
                current_pos = nextind(context, end_pos - 1)
    
                current_size_idx = 3 - current_size_idx  # Toggle between 1 and 2
            end
        end
    
        return result
    end

    function recapitalise!(text)
        text[1] = uppercase(text[1][1]) * lowercase(text[1][2:end])
        n = length(text)
        for i in 1:n
            if text[i] == "." && i+2 < n
                uppercase_word = lowercase(text[i+2])
                text[i+2] = uppercase(uppercase_word[1]) * lowercase(uppercase_word[2:end])
            end
        end
        return text
    end

    function normalize_weights(tensors, current_token)
        next_options = tensors[current_token]
        next_tokens = collect(keys(next_options))
        weights = map(w -> next_options[w], next_tokens)

        total = sum(weights)
        if total ≈ 0  # Using ≈ for floating-point comparison
            probs = fill(1.0 / length(next_tokens), length(next_tokens))
        else
            probs = weights./ total
        end
        return next_tokens, probs
    end

    function default_sampler(tensors, current_token, temperature)
        next_tokens, probs = normalize_weights(tensors, current_token)

        probs = probs.^ temperature
        probs = probs / sum(probs)
        
        r = rand()
        cumulative = 0.0
        selected_token = ""
        selected_prob = 0.0

        for (idx, token) in enumerate(next_tokens)
            cumulative += probs[idx]
            if cumulative >= r
                selected_token = token
                selected_prob = round(probs[idx], digits=2)
                break
            end
        end
        return selected_token, selected_prob
    end
end