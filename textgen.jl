using Random

text = read("sample.txt", String)
text2 = read("sample2.txt", String)

function encode(text, exclude=[" ", "(", ")", "\""], end_punctuation = [".", "!", "?"]; preserve_tokens=["'s", "'t", "'m", "'ve"])
    # Escape each preserve token to handle regex special characters
    escaped_tokens = map(t -> Regex.escape(t), preserve_tokens)
    
    # Create a regex pattern that matches word boundaries or non-word characters,
    # but treats preserve_tokens as single units
    preserve_pattern = join(escaped_tokens, "|")
    split_pattern = Regex("\\b(?:\\w+|$preserve_pattern)\\b|\\W+")
    
    # Split the text into tokens using the new pattern
    tokens = split(text, split_pattern, keepempty = false)
    
    # Remove excluded tokens
    filter!(x -> !(x in exclude), tokens)

    # Create groups for each token by their lowercase variant
    groups = Dict{String, Vector{String}}()
    for token in tokens
        lower = lowercase(token)
        if !haskey(groups, lower)
            groups[lower] = []
        end
        push!(groups[lower], token)
    end

    # Initialize the Markov dictionary and BOS token
    markov_dict = Dict{String, Dict{String, Int}}()
    init_token = "<BOS>"
    markov_dict[init_token] = Dict{String, Int}()

    # Iterate through the tokens to build the Markov chain
    for i in 1:(length(tokens) - 1)
        current_token = tokens[i]
        next_token = tokens[i+1]

        # Determine if current_token appears in more than one case
        current_lower = lowercase(current_token)
        group_size = length(groups[current_lower])
        current_base = group_size > 1 ? current_lower : current_token

        # Determine if next_token appears in more than one case
        next_lower = lowercase(next_token)
        next_group_size = length(groups[next_lower])
        next_base = next_group_size > 1 ? next_lower : next_token

        # Update BOS transitions if current token is sentence-ending punctuation
        if current_base ∈ end_punctuation
            if !haskey(markov_dict[init_token], next_base)
                markov_dict[init_token][next_base] = 0
            end
            markov_dict[init_token][next_base] += 1
        end

        # Update regular transitions
        if !haskey(markov_dict, current_base)
            markov_dict[current_base] = Dict{String, Int}()
        end

        markov_dict[current_base][next_base] += 1
    end

    return markov_dict
end

embedding = encode(text)
embedding2 = encode(text2)

function merge_embeddings(d1::Dict, d2::Dict)
    merged = deepcopy(d1)
    # Iterate over each key-value pair in the second dictionary
    for (outer_key, inner_dict) in d2
        if haskey(merged, outer_key)
            # If outer_key exists, merge the inner dictionaries
            merged_inner = merge_inner(merged[outer_key], inner_dict)
            merged[outer_key] = merged_inner
        else
            # If outer_key doesn't exist, add it
            merged[outer_key] = deepcopy(inner_dict)
        end
    end
    return merged
end

function merge_inner(i1::Dict, i2::Dict)
    merged_inner = deepcopy(i1)
    for (inner_key, value) in i2
        if haskey(merged_inner, inner_key)
            merged_inner[inner_key] += value
        else
            merged_inner[inner_key] = value
        end
    end
    return merged_inner
end

embedding = merge(embedding, embedding2)

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

function decode(embedding, max_tokens=128, stream=false, stream_rate=0)
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
        if !haskey(embedding, current_token)
            break
        end

        next_options = embedding[current_token]
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
        if selected_token ∉ [".", "!", "?", ",", "’", "'", ":", ";", "—", "/", "s", "t", "m", "ve"] && !init_token
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

Random.seed!(123)

generated_text = decode(embedding, 128, false, 1000)
println(generated_text)