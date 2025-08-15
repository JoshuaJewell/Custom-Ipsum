module Encoder
    include("types.jl")
    include("utils.jl")

    using .Types, .Utils

    export encode, sanger_encoder, default_encoder

    """
        encode(context, mode = "default"; end_punctuation = [".", "!", "?"], exclude = [" ", "(", ")", "\\"", "*"], preserve_tokens=["'s", "'t", "'m", "'ve", "'d"], fragment_size = 1)
    
    Encode the given context using the specified mode.

    ## Arguments
    - `context`: The input string to decode.
    - `mode` (optional, default: "default"): The decoding mode. Can be "default" or "sanger".
    
    ## Keyword Arguments
    - `end_punctuation` (optional, default: [".", "!", "?"]): Markers for ends of sentences. Only relevant if `mode` is "default".
    - `exclude` (optional, default: [""]: Phrases to exclude from tensordict. Value of ["\n", "(", ")", "\\"", "*"] recommended if `mode` is "default".
    - `preserve_tokens` (optional, default: [" ", "(", ")", "\\"", "*"]): Prevent tokenizer from breaking up these strings. (WIP)
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sanger".
    - `fragment_groups` (optional, default: 1): How many different fragment sizes should be parsed (high values not recommended). Only relevant if `mode` is "sanger" and `fragment_size` is specified - it's a feature, not a bug ;).
    - `verbose` (optional, default: true): Prints additional information (e.g. "Encoding in sanger mode" or "Encoded in \$s seconds")
    """
    function encode(
        context,
        mode = "default";
        end_punctuation = [".", "!", "?"], 
        exclude = [""], 
        fragment_size = 1,
        fragment_groups = 1,
        verbose = true
    )
        mode = lowercase(mode)    

        verbose && println("Encoding in $mode mode.")
        initT = time()
        
        if mode == "sanger"
            markov_dict = sanger_encoder(context, fragment_size, fragment_groups, exclude, verbose)
            args = "Fragmentation: $fragment_size by $fragment_groups."
        else
            markov_dict = default_encoder(context, end_punctuation, exclude, verbose)
            args = "Sentence enders: $end_punctuation; Preserved tokens: \$preserve_tokens."
        end

        verbose && println("\nEncoded in $(time() - initT) s")

        tensors = CompleteTensors(
            Header(mode, args),
            markov_dict,
            Dict{String, Dict{String, Float64}}(), 
            [""]
        )

        return tensors
    end

    function default_encoder(
        context,
        end_punctuation = [".", "!", "?"],
        exclude = ["\n", "(", ")", "\"", "*"],
        verbose = true
    )    
        # Extract tokens while preserving original case
        tokens = split(context, r"\b|\W+", keepempty = false)

        # Remove excluded tokens
        filter!(x -> !(x in exclude), tokens)

        # Initialize the Markov dictionary and BOS token
        markov_dict = Dict{String, Dict{String, Float64}}()
        init_token = "<BOS>"
        markov_dict[init_token] = Dict{String, Float64}()

        # Iterate through the tokens to build the Markov chain
        tokencount = length(tokens)

        for i in 1:tokencount-1
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

            progress = round(100 * i / tokencount, digits = 2)
            print("\x1b[2K\rEncoding $progress% complete. Current token: $current_token...")
        end
        verbose ? print("\x1b[2K\r100% complete.") : nothing

        return markov_dict
    end

    function sanger_encoder(
        context,
        fragment_size = 1,
        fragment_groups = 1,
        exclude = [""],
        verbose = true
    )
        context = replace(context, Regex(join(exclude, "|")) => " ")
        context = replace(context, "  " => " ")

        if fragment_size > 1
            tokens = sanger_split(context, fragment_size, fragment_groups)
        else
            fragment_size = Int(round(average_word_length(context), digits = 0))
            tokens = sanger_split(context, fragment_size, fragment_groups)
        end

        # Initialize the Markov dictionary and BOS token
        markov_dict = Dict{String, Dict{String, Float64}}()
        init_token = "<BOS>"
        pushfirst!(tokens)
        markov_dict[init_token] = Dict{String, Float64}()

        # Iterate through the tokens to build the Markov chain
        tokencount = length(tokens)

        for i in 1:tokencount-1
            current_token = tokens[i]
            next_token = tokens[i+1]
            if !haskey(markov_dict, current_token)
                markov_dict[current_token] = Dict{String, Float64}()
            end
            if !haskey(markov_dict[current_token], next_token)
                markov_dict[current_token][next_token] = 0.0
            end
            markov_dict[current_token][next_token] += 1.0

            progress = round(100 * i / tokencount, digits = 2)

            if i == 1 || endswith(current_token, "\n")
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