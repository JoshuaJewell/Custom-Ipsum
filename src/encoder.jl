module Encoder
    include("helpers.jl")

    using .Helpers

    export encode

    """
        encode(context, mode = "default"; end_punctuation = [".", "!", "?"], exclude = [" ", "(", ")", "\"", "*"], preserve_tokens=["'s", "'t", "'m", "'ve", "'d"], fragment_size = 1)
    
    Encode the given context using the specified mode.

    ## Arguments
    - `context`: The input string to decode.
    - `mode` (optional, default: "default"): The decoding mode. Can be "default" or "sanger".
    
    ## Keyword Arguments
    - `end_punctuation` (optional, default: [".", "!", "?"]): Markers for ends of sentences. Only relevant if `mode` is "default".
    - `exclude` (optional, default: [" ", "(", ")", "\"", "*"]): Tokens to exclude from tensordict. Only relevant if `mode` is "default".
    - `preserve_tokens` (optional, default: [" ", "(", ")", "\"", "*"]): Prevent tokenizer from breaking up these strings. (WIP)
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sagner".
    """
    function encode(context, mode = "default"; end_punctuation = [".", "!", "?"], exclude = [" ", "(", ")", "\"", "*"], preserve_tokens=["'s", "'t", "'m", "'ve", "'d"], fragment_size = 1)
        initT = time()
        mode = lowercase(mode)
        
        if mode == "sanger"
            sanger_encoder(context, fragment_size = 1)
        else
            default_encoder(context, end_punctuation, exclude, preserve_tokens)
        end

        println("\nEncoded in $(time() - initT) s")
    end

    function default_encoder(context; end_punctuation = [".", "!", "?"], exclude=[" ", "(", ")", "\"", "*"], preserve_tokens=["'s", "'t", "'m", "'ve"])    
        # Extract tokens while preserving original case
        tokens = split(context, r"\b|\W+", keepempty = false)

        # Remove excluded tokens
        filter!(x -> !(x in exclude), tokens)

        # Initialize the Markov dictionary and BOS token
        markov_dict = Dict{String, Dict{String, Float64}}()
        init_token = "<BOS>"
        markov_dict[init_token] = Dict{String, Float64}()

        # Iterate through the tokens to build the Markov chain
        for i in 1:length(tokens)-1
            current_token = tokens[i]
            next_token = tokens[i+1]
            if !haskey(markov_dict, current_token)
                markov_dict[current_token] = Dict{String, Float64}()
            end
            if !haskey(markov_dict[current_token], next_token)
                markov_dict[current_token][next_token] = 0.0
            end
            markov_dict[current_token][next_token] += 1.0

            if current_token in end_punctuation
                markov_dict[init_token][next_token] = get(markov_dict[init_token], next_token, 0) + 1
            else
                if !haskey(markov_dict, current_token)
                    markov_dict[current_token] = Dict{String, Float64}()
                end
                markov_dict[current_token][next_token] = get(markov_dict[current_token], next_token, 0) + 1
            end

            progress = round(100 * i / length(tokens), digits = 2)
            print("\x1b[2K\r$progress% complete. Current token: $current_token...")
        end

        return markov_dict
    end

    function sanger_encoder(context; fragment_size=1)
        if fragment_size > 1
            tokens = sanger_split(context, fragment_size)
        else
            fragment_size = round(average_word_length(context), digits = 0)
            tokens = sanger_split(context, fragment_size)
        end

        # Initialize the Markov dictionary and BOS token
        markov_dict = Dict{String, Dict{String, Float64}}()
        init_token = "<BOS>"
        markov_dict[init_token] = Dict{String, Float64}()

        # Iterate through the tokens to build the Markov chain
        for i in 1:length(tokens)-1
            current_token = tokens[i]
            next_token = tokens[i+1]
            if !haskey(markov_dict, current_token)
                markov_dict[current_token] = Dict{String, Float64}()
            end
            if !haskey(markov_dict[current_token], next_token)
                markov_dict[current_token][next_token] = 0.0
            end
            markov_dict[current_token][next_token] += 1.0

            progress = round(100 * i / length(tokens), digits = 2)
            print("\x1b[2K\r$progress% complete. Current token: $(filter(c -> c != '\n', current_token))...")
            flush(stdout)

            if current_token in ["."]
                markov_dict[init_token][next_token] = get(markov_dict[init_token], next_token, 0) + 1
            else
                if !haskey(markov_dict, current_token)
                    markov_dict[current_token] = Dict{String, Float64}()
                end
                markov_dict[current_token][next_token] = get(markov_dict[current_token], next_token, 0) + 1
            end
        
        end

        return markov_dict
    end
end