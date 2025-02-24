module Decoder
    include("types.jl")
    include("utils.jl")

    using .Types, .Utils 

    export decode

    """
        decode(tensors, mode = "default"; max_tokens = 128, beam_width = 3, stream = false, stream_rate = 1000)

    Decode the given tensors using the specified mode.

    ## Arguments
    - `tensors`: The input tensors to decode.
    
    ## Keyword Arguments
    - `max_tokens` (optional, default: 128): Maximum number of tokens to decode.
    - `beam_width` (optional, default: 3): Beam width for beam search decoding. Only relevant if `mode` is "beamsearch".
    - `stream` (optional, default: false): Whether to stream the output. Only relevant if `mode` is "sanger" or "default".
    - `stream_rate` (optional, default: 1000): How quickly to stream output. Set to 0 for infinite. Only relevant if `mode` is "sanger" or "default".
    """
    function decode(
        tensors;
        max_tokens = 128,
        beam_width = 3,
        stream = false,
        stream_rate = 1000
    )
        initT = time()

        mode = tensors.header.encoding_method

        println("Decoding in $(mode) mode.")
        if mode == "sanger"
            output = sanger_decoder(tensors.forward_markov, max_tokens, stream, stream_rate)
        elseif mode == "beamsearch"
            output = beam_search_decoder(tensors.forward_markov, max_tokens, beam_width)
        else
            output = default_decoder(tensors.forward_markov, max_tokens, stream, stream_rate)
        end
        
        println("\nDecoded in $(time() - initT) s")
        
        if !stream
            return output
        end
    end

    function default_decoder(
        tensors,
        max_tokens=128,
        stream=false,
        stream_rate=0
    )
        if max_tokens == 0
            return ""
        end

        text = []
        current_token = "<BOS>"
        init_token = true

        if stream
            capitalise_next = true
            if stream_rate > 0
                stream_rate = 1 / stream_rate
            end
        end

        for i in 2:max_tokens
            if !haskey(tensors, current_token)
                break
            end

            next_options = tensors[current_token]
            next_tokens = collect(keys(next_options))
            weights = map(w -> next_options[w], next_tokens)

            # Normalize weights to probabilities
            total = sum(weights)
            if total == 0
                probs = fill(1.0 / max_tokens(next_tokens), max_tokens(next_tokens))
            else
                probs = weights ./ total
            end

            # Generate a random number and select the next token based on cumulative probabilities
            r = rand()
            cumulative = 0.0
            selected_token = ""
            for (idx, token) in enumerate(next_tokens)
                cumulative += probs[idx]
                if cumulative >= r
                    selected_token = token
                    break
                end
            end

            # Add space unless it's punctuation
            if selected_token ∉ [".", "!", "?", ",", "’", "'", ":", ";", "—", "/", "s", "t", "m", "d", "ve", "…"] && !init_token
                push!(text, " ")
                if stream
                    print(" ")
                    flush(stdout)
                end
            end

            current_token = selected_token
            push!(text, current_token)

            if stream
                if capitalise_next
                    stream_token = uppercase(current_token[1]) * lowercase(current_token[2:end])
                    capitalise_next = false
                else
                    stream_token = current_token
                end

                if init_token == true
                    stream_token = uppercase(stream_token[1]) * lowercase(stream_token[2:end])
                end

                if current_token ∈ [".", "!", "?"]
                    capitalise_next = true
                end

                if stream_token !== nothing
                    sleep(stream_rate)
                    print(stream_token)
                    flush(stdout)
                end
            end
            init_token = false
        end
        if !stream
            recapitalise!(text)
            return join(text)
        end
    end

    function sanger_decoder(tensors, max_tokens=128, stream=false, stream_rate=0)
        if max_tokens == 0
            return ""
        end

        text = []
        current_token = "<BOS>"
        init_token = true

        if stream
            if stream_rate > 0
                stream_rate = 1 / stream_rate
            end
        end

        for i in 2:max_tokens
            if !haskey(tensors, current_token)
                break
            end

            next_options = tensors[current_token]
            next_tokens = collect(keys(next_options))
            weights = map(w -> next_options[w], next_tokens)

            # Normalize weights to probabilities
            total = sum(weights)
            if total == 0
                probs = fill(1.0 / length(next_tokens), length(next_tokens))
            else
                probs = weights ./ total
            end

            # Generate a random number and select the next token based on cumulative probabilities
            r = rand()
            cumulative = 0.0
            selected_token = ""
            for (idx, token) in enumerate(next_tokens)
                cumulative += probs[idx]
                if cumulative >= r
                    selected_token = token
                    break
                end
            end

            current_token = selected_token
            push!(text, current_token)

            if stream
                stream_token = current_token

                if init_token == true
                    stream_token = uppercase(stream_token[1]) * lowercase(stream_token[2:end])
                end

                if current_token ∈ [".", "!", "?"]
                    capitalise_next = true
                end

                if stream_token !== nothing
                    sleep(stream_rate)
                    print(stream_token)
                    flush(stdout)
                end
            end
            init_token = false
        end
        if !stream
            return join(text)
        end
    end

    function beam_search_decoder(
        tensors,
        max_tokens=128,
        beam_width=3
    )
        if max_tokens == 0
            return ""
        end

        beam = [Dict(:sequence => ["<BOS>"], :weight => 1.0)]

        for i in 2:max_tokens
            # Expand the current beam
            new_beam = []

            for candidate in beam
                current_sequence = candidate[:sequence]
                current_weight = candidate[:weight]
                last_token = current_sequence[end]

                # If the last token has no next options, continue with the current sequence
                if !haskey(tensors, last_token)
                    push!(new_beam, Dict(:sequence => deepcopy(current_sequence), :weight => current_weight))
                    continue
                end

                # Get available next tokens and their weights
                next_options = tensors[last_token]
                next_tokens = collect(keys(next_options))
                weights = map(w -> next_options[w], next_tokens)

                # Normalize weights to probabilities
                total = sum(weights)
                if total == 0
                    probs = fill(1.0 / length(next_tokens), length(next_tokens))
                else
                    probs = weights ./ total
                end

                # Extend each possible next token as a new candidate
                for (idx, token) in enumerate(next_tokens)
                    new_sequence = deepcopy(current_sequence)
                    push!(new_sequence, token)
                    new_weight = current_weight * probs[idx]
                    push!(new_beam, Dict(:sequence => new_sequence, :weight => new_weight))
                end
            end

            # Prune the beam by keeping only the top 'beam_width' candidates
            if !isempty(new_beam)
                # Sort by descending weight
                sort!(new_beam, by = x -> -x[:weight])
                # Keep only top 'beam_width' candidates
                beam = new_beam[1:min(beam_width, length(new_beam))]
            else
                # If no possible candidates, break early
                break
            end
        end

        # Select the best sequence (highest weight)
        best_sequence = beam[1][:sequence]
        text = best_sequence[2:end]  # Exclude the initial <BOS> token

        # Perform recapitalization and punctuation adjustments
        recapitalise!(text)

        return join(text, " ")
    end
end