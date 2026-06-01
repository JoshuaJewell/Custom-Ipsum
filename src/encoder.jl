module Encoder
    using ..Types, ..Utils

    export encode

    """
        encode(context, mode = "sanger"; end_punctuation = [".", "!", "?"], exclude = [" ", "(", ")", "\\"", "*"], preserve_tokens=["'s", "'t", "'m", "'ve", "'d"], fragment_size = 1)

    Encode the given context using the specified mode.

    ## Arguments
    - `context`: The input string to decode.
    - `mode` (optional, default: "sanger"): The encoding mode. Only "sanger" is currently supported; "default" is legacy and raises an informative error pending a rewrite onto the numeric model.
    
    ## Keyword Arguments
    - `end_punctuation` (optional, default: [".", "!", "?"]): Markers for ends of sentences. Only relevant if `mode` is "default".
    - `exclude` (optional, default: [""]: Phrases to exclude from tensordict. Value of ["\\n", "(", ")", "\\"", "*"]. Only relevant if `mode` is "default".
    - `preserve_tokens` (optional, default: [" ", "(", ")", "\\"", "*"]): Prevent tokenizer from breaking up these strings. (WIP)
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sanger".
    - `fragment_groups` (optional, default: 1): How many different fragment sizes should be parsed (high values not recommended). Only relevant if `mode` is "sanger" and `fragment_size` is specified - it's a feature, not a bug ;).
    - `verbose` (optional, default: true): Prints additional information (e.g. "Encoding in sanger mode" or "Encoded in \$s seconds")
    """
    function encode(
        context,
        mode = "sanger";
        end_punctuation = [".", "!", "?"],
        exclude = String[],
        fragment_size = 1,
        fragment_groups = 1,
        verbose = true
    )
        mode = lowercase(mode)

        # Only "sanger" is currently supported. The "default" encoder below is
        # legacy: it produces a string-keyed Dict that the numeric CompleteTensors
        # type can no longer hold. Guard here rather than fail cryptically later.
        if mode != "sanger"
            error("Encoder mode '$mode' is not supported; only \"sanger\" is. " *
                  "(The \"default\" path is legacy, pending a rewrite onto the numeric model.)")
        end

        verbose && println("Encoding in $mode mode.")
        initT = time()

        markov_dict, token_to_id, id_to_token = sanger_encoder(context, fragment_size, fragment_groups, verbose)
        args = "Fragmentation: $fragment_size by $fragment_groups."

        verbose && println("\nEncoded in $(time() - initT) s")

        tensors = pack_ctensors(mode, args, markov_dict, token_to_id, id_to_token)

        return tensors
    end

    ## LEGACY
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
        verbose = true
    )
        #!isempty(exclude) ? context = replace(context, Regex(join(exclude, "|")) => " ") : nothing
        #context = replace(context, "  " => " ")

        if fragment_size > 1
            tokens = sanger_split(context, fragment_size, fragment_groups)
        else
            fragment_size = Int(round(average_word_length(context), digits = 0))
            tokens = sanger_split(context, fragment_size, fragment_groups)
        end

        # Build vocabulary
        unique_tokens = unique(tokens)
        pushfirst!(unique_tokens, "<BOS>")  # Ensure BOS is index 1
        
        token_to_id = Dict(token => i for (i, token) in enumerate(unique_tokens))
        id_to_token = unique_tokens
        
        # Build numeric Markov chain
        markov_dict = Dict{Int, Dict{Int, Float64}}()
        bos_id = token_to_id["<BOS>"]
        markov_dict[bos_id] = Dict{Int, Float64}()
        
        tokencount = length(tokens)
        for i in 1:tokencount-1
            current_id = token_to_id[tokens[i]]
            next_id = token_to_id[tokens[i+1]]
            
            if !haskey(markov_dict, current_id)
                markov_dict[current_id] = Dict{Int, Float64}()
            end
            markov_dict[current_id][next_id] = get(markov_dict[current_id], next_id, 0.0) + 1.0
            
            # Handle BOS transitions
            if i == 1 || endswith(tokens[i], "\n")
                markov_dict[bos_id][next_id] = get(markov_dict[bos_id], next_id, 0.0) + 1.0
            end
        end

        return markov_dict, token_to_id, id_to_token
    end
end