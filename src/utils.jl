module Utils

    export average_word_length, sanger_split, recapitalise!

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

    function sanger_split(
        context,
        fragment_size,
        fragment_groups = 1
    )
        result = []

        for group in 1:fragment_groups
            push!(result, "\n")
            group_array = sanger_split_base(context, fragment_size)
            append!(result, group_array)
            fragment_size += group
        end

        return result
    end

    function sanger_split_base(
        context,
        fragment_size
    )
        n = ncodeunits(context)  # Use ncodeunits to get the number of code units
        result = Vector{String}()

        for offset::Int64 in 1:fragment_size
            current_pos = offset
            while current_pos <= n
                # Calculate the end position of the fragment
                end_pos = current_pos
                for _ in 1:(fragment_size - 1)
                    if end_pos > n
                        break
                    end
                    end_pos = nextind(context, end_pos)
                end

                # Extract the fragment
                token = context[current_pos:prevind(context, end_pos)]
                push!(result, token) 
                
                # Move to the next starting position
                current_pos = nextind(context, end_pos - 1)
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
end