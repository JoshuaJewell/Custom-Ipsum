module Decoder
    using ..Types, ..Utils

    export decode

    """
        decode(tensors, mode = "default"; max_tokens = 128, beam_width = 3, stream = false, stream_rate = 1000)

    Decode the given tensors using the specified mode.

    ## Arguments
    - `tensors`: The input tensors to decode from.
    
    ## Keyword Arguments
    - `max_tokens` (optional, default: 128): Maximum number of tokens to decode.
    - `beam_width` (optional, default: 3): Beam width for beam search decoding. Only relevant if `mode` is "beamsearch".
    - `stream` (optional, default: false): Whether to stream the output. Only relevant if `mode` is "sanger" or "default".
    - `stream_rate` (optional, default: 1000): How quickly to stream output. Set to 0 for infinite. Only relevant if `mode` is "sanger" or "default".
    - `show_tokens` (optional, default: false): Marks tokens in output, will also print token probability if `stream` is true. Only relevant if `mode` is "sanger".
    - `temperature` (optional, default: 1): Sets the temperature of the sampler, higher values increase diversity and lower values increase confidence. Only relevant if `mode` is "sanger" or "default".
    """
    function decode(
        tensors;
        max_tokens = 128,
        beam_width = 3,
        stream = false,
        stream_rate = 1000,
        show_tokens = false,
        temperature = 1
    )
        initT = time()

        mode = tensors.header.encoding_method

        # Only the numeric "sanger" model is currently supported. The "default"
        # and "beamsearch" decoders below are legacy, kept against the abandoned
        # string-keyed API; guard here so an old model fails clearly rather than
        # crashing cryptically deep inside an unsupported path.
        if mode != "sanger"
            error("Decoding mode '$mode' is not supported; only \"sanger\" models can be decoded. " *
                  "(\"default\" and \"beamsearch\" are legacy, pending a rewrite onto the numeric model.)")
        end

        println("Decoding in $(mode) mode.")
        output = sanger_decoder(tensors, max_tokens, stream, stream_rate, show_tokens, temperature)

        println("\nDecoded in $(time() - initT) s")
        
        if !stream
            return output
        end
    end

    ## LEGACY
    function default_decoder(tensors, max_tokens=128, stream=false, stream_rate=0, temperature=1)
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

            selected_token, prob = default_sampler(tensors, current_token, temperature)

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

    function sanger_decoder(tensors, max_tokens=128, stream=false, stream_rate=0, show_tokens=false, temperature=1)
        # A model with interior tables walks with a delimiter stack; an empty-
        # interiors model takes the flat path below, byte-for-byte as before.
        isempty(tensors.interiors) || return nested_decoder(tensors, max_tokens, stream, stream_rate, show_tokens, temperature)
        if max_tokens == 0
            return ""
        end
        
        text = []
        current_id = tensors.vocabulary.token_to_id["<BOS>"]

        # Treat stream_rate as tokens per second; 0 means stream with no delay.
        delay = (stream && stream_rate > 0) ? 1 / stream_rate : 0

        for i in 2:max_tokens
            step = sample_next(tensors.forward_markov, current_id, temperature)
            step === nothing && break

            selected_id, prob = step
            selected_token = tensors.vocabulary.id_to_token[selected_id]

            current_id = selected_id
            push!(text, selected_token)

            if stream
                delay > 0 && sleep(delay)
                if show_tokens
                    print(selected_token, "\$$(prob)")
                else
                    print(selected_token)
                end
                flush(stdout)
            end
        end

        return stream ? nothing : join(text)
    end

    # The close glyph emitted when a class is force-closed at an interior
    # dead-end. Parentheses close with ')'; the quote class with '"'.
    const CLASS_CLOSE = Dict{Symbol,Char}(:paren => ')', :quote => '"')

    # Stack-aware walk. The active table is the outer chain at depth zero, else
    # the interior chain named by the top of the stack. After each token its
    # characters update the stack via the shared classifier; a token-final quote,
    # whose following character is not yet known, is decided by stack parity. At
    # depth zero a dead-end ends the text; inside an interior a dead-end closes
    # one level (emitting that close glyph) and retries. The truncation tail at
    # max_tokens is left exactly as it falls, as an unfinished sentence would be.
    function nested_decoder(tensors, max_tokens=128, stream=false, stream_rate=0, show_tokens=false, temperature=1)
        if max_tokens == 0
            return ""
        end

        text = []
        outer = tensors.forward_markov
        interiors = tensors.interiors
        active = Set(keys(interiors))
        current_id = tensors.vocabulary.token_to_id["<BOS>"]
        stack = Symbol[]
        prevchar = '\0'

        delay = (stream && stream_rate > 0) ? 1 / stream_rate : 0

        table() = isempty(stack) ? outer : interiors[stack[end]]

        # Record a token in `text`; when streaming, print `display` (which carries
        # the probability annotation under show_tokens) rather than the raw token.
        function emit(s, display = s)
            push!(text, s)
            if stream
                delay > 0 && sleep(delay)
                print(display)
                flush(stdout)
            end
        end

        for i in 2:max_tokens
            # Advance the active table; on exhaustion, close one interior level
            # and retry, or stop at depth zero.
            step = sample_next(table(), current_id, temperature)
            dead = false
            while step === nothing
                if isempty(stack)
                    dead = true
                    break
                end
                cls = pop!(stack)
                g = CLASS_CLOSE[cls]
                emit(string(g))
                prevchar = g
                step = sample_next(table(), current_id, temperature)
            end
            dead && break

            selected_id, prob = step
            selected_token = tensors.vocabulary.id_to_token[selected_id]
            current_id = selected_id

            display = show_tokens ? selected_token * "\$$(prob)" : selected_token
            emit(selected_token, display)

            # Update the stack from the emitted token's characters. Collect to a
            # Vector{Char} so the classifier can look one character ahead by index.
            cs = collect(selected_token)
            for j in eachindex(cs)
                g = cs[j]
                nxt = j < length(cs) ? cs[j+1] : '\0'
                if g == '"' && nxt == '\0'
                    # Token-final quote: the following character is not yet known.
                    # Close an open quote (parity); otherwise open only when the
                    # preceding character presents as an opening boundary, matching
                    # the encoder's flanking rule. A quote hard against a letter
                    # with nothing open is a stray close and is left alone.
                    event, _ = delimiter_event(prevchar, g, '\0')
                    if !isempty(stack) && stack[end] == :quote
                        pop!(stack)
                    elseif event != :close && :quote in active
                        push!(stack, :quote)
                    end
                else
                    event, class = delimiter_event(prevchar, g, nxt)
                    if class in active
                        if event == :open
                            push!(stack, class)
                        elseif event == :close && !isempty(stack) && stack[end] == class
                            pop!(stack)
                        end
                    end
                end
                prevchar = g
            end
        end

        return stream ? nothing : join(text)
    end

    ## LEGACY (unsupported)
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

                next_tokens, probs = normalize_weights(tensors, current_token)

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