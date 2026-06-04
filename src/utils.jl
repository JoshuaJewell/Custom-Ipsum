module Utils
    using ..Types

    export average_word_length, sanger_split, recapitalise!, normalize_weights, default_sampler, merge_tensordicts, pack_ctensors, unpack_ctensors, merge_complete_tensors, delimiter_event, scan_depth_classes

    # Characters that bound a quote: a missing neighbour, whitespace, brackets,
    # and the punctuation a closing quote may sit against.
    const QUOTE_BOUNDS = Set{Char}(['(', ')', '.', ',', ';', ':', '!', '?'])
    isbound(c::Char) = c == '\0' || isspace(c) || c in QUOTE_BOUNDS

    # Classify one delimiter occurrence, returning (event, class) with event in
    # (:open, :close, :none) and class in (:paren, :quote, :none). Parentheses
    # are directional. A double quote opens when its outer (preceding) side is a
    # boundary and its inner (following) side is not, closes in the mirror case,
    # and reads as an apostrophe (:none) when letters flank it. '\0' marks a
    # missing neighbour at the start or end of the stream.
    function delimiter_event(prev::Char, glyph::Char, nxt::Char)
        glyph == '(' && return (:open, :paren)
        glyph == ')' && return (:close, :paren)
        if glyph == '"'
            pb = isbound(prev)
            nb = isbound(nxt)
            pb && !nb && return (:open, :quote)
            !pb && nb && return (:close, :quote)
            return (:none, :none)
        end
        return (:none, :none)
    end

    # For each character position, the delimiter class in force just AFTER that
    # character: :outer at depth zero, else the class on top of the stack. Only
    # the classes named in `active` are treated as delimiters, so an empty
    # `active` yields all :outer (the flat model). An unmatched close is ignored,
    # for robustness on real-world text.
    function scan_depth_classes(context::AbstractString, active::AbstractSet{Symbol})
        cs = collect(context)
        n = length(cs)
        classes = Vector{Symbol}(undef, n)
        stack = Symbol[]
        for i in 1:n
            prev = i > 1 ? cs[i-1] : '\0'
            nxt = i < n ? cs[i+1] : '\0'
            event, class = delimiter_event(prev, cs[i], nxt)
            if class in active
                if event == :open
                    push!(stack, class)
                elseif event == :close && !isempty(stack) && stack[end] == class
                    pop!(stack)
                end
            end
            classes[i] = isempty(stack) ? :outer : stack[end]
        end
        return classes
    end

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
        tokens = String[]
        spans = Tuple{Int,Int}[]

        # Regular split: associations between fragments of the same group.
        for group in 1:fragment_groups
            push!(tokens, "\n"); push!(spans, (0, 0))
            toks, sps = sanger_split_base(context, fragment_size)
            append!(tokens, toks); append!(spans, sps)
            fragment_size += 1
        end

        # Alternating split: associations between fragments of different groups.
        sizes = fragment_size:(fragment_size + fragment_groups - 1)
        if fragment_groups > 1
            for size1 in sizes
                for size2 in sizes
                    if size1 != size2
                        toks, sps = sanger_split_alt(context, size1, size2)
                        append!(tokens, toks); append!(spans, sps)
                    end
                end
            end
        end

        return tokens, spans
    end

    function sanger_split_base(context, fragment_size)
        tokens = String[]
        spans = Tuple{Int,Int}[]
        chars = collect(context)

        for offset in 1:fragment_size
            current_pos = offset
            while current_pos <= length(chars)
                end_pos = current_pos
                for _ in 1:(fragment_size - 1)
                    if end_pos >= length(chars)
                        break
                    end
                    end_pos += 1
                end

                if end_pos > length(chars)
                    break
                end

                push!(tokens, join(chars[current_pos:end_pos]))
                push!(spans, (current_pos, end_pos))
                current_pos += fragment_size
            end
        end

        return tokens, spans
    end

    function sanger_split_alt(context, size1, size2)
        tokens = String[]
        spans = Tuple{Int,Int}[]
        chars = collect(context)
        n = length(chars)
        sizes = [size1, size2]

        for offset in 1:min(size1, size2)
            current_pos = offset - 1
            current_size_idx = 1

            while current_pos < n
                chunk_size = sizes[current_size_idx]
                end_pos = min(current_pos + chunk_size - 1, n - 1)
                push!(tokens, join(chars[(current_pos + 1):(end_pos + 1)]))
                push!(spans, (current_pos + 1, end_pos + 1))
                current_pos += chunk_size
                current_size_idx = 3 - current_size_idx
            end
        end

        return tokens, spans
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

    #function pack_ctensors(encoding_method = "unknown", metadata = "", forward_markov = Dict{String, Dict{String, Float64}}(), reverse_markov = Dict{String, Dict{String, Float64}}(), token_index = [""])
    #    ctensors = CompleteTensors(
    #        Header(encoding_method, metadata),
    #        forward_markov,
    #        reverse_markov, 
    #        token_index
    #    )

    #    return ctensors
    #end

    function pack_ctensors(encoding_method = "unknown", metadata = "", forward_markov = Dict{Int, Dict{Int, Float64}}(), token_to_id = Dict{String, Int}(), id_to_token = Vector{String}(), interiors = Dict{Symbol, Dict{Int, Dict{Int, Float64}}}())
        ctensors = CompleteTensors(
            Header(encoding_method, metadata),
            Vocabulary(token_to_id, id_to_token),
            forward_markov,
            interiors
        )

        return ctensors
    end

    ##############################################

    function merge_complete_tensors(header, markov1, markov2, id_to_token1, id_to_token2)
        # Step 1: Count token frequencies
        # Float64, not Int: weights may be fractional once merge_mult scales them.
        token_frequency = Dict{String, Float64}()

        function count_frequencies(forward_markov, id_to_token)
            for (id, transitions) in forward_markov
                token = id_to_token[id]
                token_frequency[token] = get(token_frequency, token, 0) + sum(values(transitions))
            end
        end

        count_frequencies(markov1, id_to_token1)
        count_frequencies(markov2, id_to_token2)

        # Step 2: Create a combined vocabulary sorted by frequency
        combined_tokens = Set{String}()
        for token in keys(token_frequency)
            push!(combined_tokens, token)
        end

        sorted_tokens = sort(collect(combined_tokens), by = x -> -get(token_frequency, x, 0))

        # Create new vocabulary
        new_token_to_id = Dict{String, Int}()
        new_id_to_token = Vector{String}()
        for (id, token) in enumerate(sorted_tokens)
            new_token_to_id[token] = id
            push!(new_id_to_token, token)
        end
        new_vocabulary = Vocabulary(new_token_to_id, new_id_to_token)

        # Step 3: Remap forward_markov dictionaries
        new_forward_markov = Dict{Int, Dict{Int, Float64}}()

        function remap_ids(forward_markov::Dict{Int, Dict{Int, Float64}}, id_to_token)
            remapped = Dict{Int, Dict{Int, Float64}}()
            for (old_id, transitions) in forward_markov
                token = id_to_token[old_id]
                new_id = new_token_to_id[token]
                remapped[new_id] = Dict{Int, Float64}()
                for (next_id, prob) in transitions
                    next_token = id_to_token[next_id]
                    new_next_id = new_token_to_id[next_token]
                    remapped[new_id][new_next_id] = get(remapped[new_id], new_next_id, 0.0) + prob
                end
            end
            return remapped
        end

        remapped_markov1 = remap_ids(markov1, id_to_token1)
        remapped_markov2 = remap_ids(markov2, id_to_token2)

        # Step 4: Merge the remapped forward_markov dictionaries
        for (id, transitions) in remapped_markov1
            if haskey(new_forward_markov, id)
                for (next_id, prob) in transitions
                    new_forward_markov[id][next_id] = get(new_forward_markov[id], next_id, 0.0) + prob
                end
            else
                new_forward_markov[id] = transitions
            end
        end

        for (id, transitions) in remapped_markov2
            if haskey(new_forward_markov, id)
                for (next_id, prob) in transitions
                    new_forward_markov[id][next_id] = get(new_forward_markov[id], next_id, 0.0) + prob
                end
            else
                new_forward_markov[id] = transitions
            end
        end

        # Create the merged CompleteTensors
        return CompleteTensors(Header(header.encoding_method,header.metadata), new_vocabulary, new_forward_markov)
    end

end