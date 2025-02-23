using JLD2, Random

function encode(context, exclude=[" ", "(", ")", "\"", "*"], end_punctuation = [".", "!", "?"]; preserve_tokens=["'s", "'t", "'m", "'ve"])    
    # Extract tokens while preserving original case and keeping preserve_tokens intact
    tokens = split(context, r"\b|\W+", keepempty = false)

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
    markov_dict = Dict{String, Dict{String, Float64}}()
    init_token = "<BOS>"
    markov_dict[init_token] = Dict{String, Float64}()

    # Iterate through the tokens to build the Markov chain
    for i in 1:(length(tokens) - 1)
        current_token = tokens[i]
        next_token = tokens[i+1]

        # Determine if current_token appears in more than one case
        current_lower = lowercase(current_token)
        group_size = length(get(groups, current_lower, []))
        current_base = group_size > 1 ? current_lower : current_token

        # Determine if next_token appears in more than one case
        next_lower = lowercase(next_token)
        next_group_size = length(get(groups, next_lower, []))
        next_base = next_group_size > 1 ? next_lower : next_token

        # Update BOS transitions if current token is sentence-ending punctuation
        if current_base ∈ end_punctuation
            markov_dict[init_token][next_base] = get(markov_dict[init_token], next_base, 0) + 1
        end

        # Update regular transitions
        if !haskey(markov_dict, current_base)
            markov_dict[current_base] = Dict{String, Int}()
        end

        markov_dict[current_base][next_base] = get(markov_dict[current_base], next_base, 0) + 1
    end

    return markov_dict
end

function merge_inner(d1::Dict{String, Float64}, d2::Dict{String, Float64}, ratio::Float64)
    merged_inner = Dict{String, Float64}()

    # Get all unique keys from both dictionaries
    all_keys = union(keys(d1), keys(d2))

    for key in all_keys
        val1 = get(d1, key, 0.0)
        val2 = get(d2, key, 0.0)

        # Compute the weighted sum
        new_val = val1 * (1 - ratio) + val2 * ratio

        # Assign the new value only if it's not zero
        if new_val != 0.0
            merged_inner[key] = new_val
        end
    end

    return merged_inner
end

# Function to merge two dictionaries with the given ratio
function merge_embeddings(d1::Dict{String, Dict{String, Float64}}, d2::Dict{String, Dict{String, Float64}}, ratio::Float64)
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

contexts = []
for i in 1:12
    context = read("sample$i.txt", String)
    push!(contexts, context)
end

embeddings = [encode(context) for context in contexts]

merged_embedding = embeddings[1]

# Use `eachindex` to loop over the indices safely
for i in eachindex(embeddings[2:end])
    ratio = 1.0 / i
    global merged_embedding = merge_embeddings(merged_embedding, embeddings[i + 1], ratio)
end

jldopen("sample.embedding", "w") do file
    file["data"] = merged_embedding
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
        if selected_token ∉ [".", "!", "?", ",", "’", "'", ":", ";", "—", "/", "s", "t", "m", "ve", "…"] && !init_token
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

sampleembedding = jldopen("sample.embedding", "r") do file
    file["data"]
end

generated_text = decode(sampleembedding, 128, false, 1000)
println(generated_text)