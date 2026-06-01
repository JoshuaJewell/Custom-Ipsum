module Tools
    using ..Types, ..Encoder, ..Decoder, ..Utils
    using Distributed

    export encode_multiple, encoder_decoder, merge_ctensors

    function encoder_decoder(
        context,
        mode = "sanger";
        end_punctuation = [".", "!", "?"], 
        exclude = [" ", "(", ")", "\"", "*"], 
        fragment_size = 1,
        fragment_groups = 1,
        max_tokens = 128,
        stream = false,
        stream_rate = 1000,
        show_tokens = false,
        temperature = 1
    )
        return decode(
            encode(context, mode, end_punctuation=end_punctuation, exclude=exclude, fragment_size=fragment_size, fragment_groups=fragment_groups),
            max_tokens=max_tokens,
            stream=stream,
            stream_rate=stream_rate,
            show_tokens=show_tokens,
            temperature=temperature
    )
    end

    """
        function encode_multiple(path_to_context = "../data/contexts/", encoder_mode = "sanger", fragment_size = 1, fragment_groups = 1)
    
    Encodes multiple contexts from a set of files and merges the weights. Recommended to store contexts in format:
        context1.txt
        context2.txt
        etc...
    But will read all files in dir.

    ## Arguments
    - `path_to_context` (default: "../data/contexts/"): Where your context files to encode and merge are.
    - `context_filename` (default: "context"): The unchanged part of your contexts' filenames.
    - `context_file_no` (default: 2): The number of context files, indexed from 1.

    ## Keyword Arguments
    - `encoder_mode` (optional, default: "default"): The encoder mode. Can be "default", or "sanger".
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sanger".
    - `fragment_groups` (optional, default: 1): How many different fragment sizes should be parsed (high values not recommended). Only relevant if `mode` is "sanger" and `fragment_size` is specified.

    """
    function encode_multiple(
        path_to_context = "../data/contexts/";
        encoder_mode = "sanger",
        fragment_size = 1,
        fragment_groups = 1
    )
        encoder_mode = lowercase(encoder_mode)

        contexts = []
        files = readdir(path_to_context)

        for file in files
            # Read each file's content
            context = read(joinpath(path_to_context, file), String)
            push!(contexts, context)
        end

        println()
        contextcount = length(contexts)
        args = ""

        num_threads = Threads.nthreads()
        results = Vector{Any}(undef, num_threads)
        result_channel = Channel{Any}(num_threads)

        final_task = @async begin
            final_merged_tensors = nothing
            finished_count = 0
            while finished_count < num_threads
                result = take!(result_channel)
                print("\x1b[2K\rFinalising: $finished_count of $(length(results))")
                if result !== nothing
                    if final_merged_tensors === nothing
                        final_merged_tensors = result
                    else
                        final_merged_tensors = merge_complete_tensors(final_merged_tensors.header,
                            final_merged_tensors.forward_markov, result.forward_markov,
                            final_merged_tensors.vocabulary.id_to_token, result.vocabulary.id_to_token)
                    end
                end
                finished_count += 1
            end

            tensors = final_merged_tensors

            return tensors
        end

        Threads.@threads for tid in 1:num_threads
            start = ((tid - 1) * div(contextcount, num_threads)) + 1
            stop = min(tid * div(contextcount, num_threads), contextcount)
            if tid == num_threads
                stop = contextcount
            end

            print("\x1b[5A\rFour threads sample:\x1b[5B")

            local_tensor = nothing

            for i in start:stop
                context = contexts[i]

                (Threads.threadid() == 1) ? print("\x1b[4A\rThread 1 encoding file $(i-start+1) of $(stop-start+1)...\x1b[4B") : nothing
                (Threads.threadid() == 2) ? print("\x1b[3A\rThread 2 encoding file $(i-start+1) of $(stop-start+1)...\x1b[3B") : nothing
                (Threads.threadid() == 3) ? print("\x1b[2A\rThread 3 encoding file $(i-start+1) of $(stop-start+1)...\x1b[2B") : nothing
                (Threads.threadid() == 4) ? print("\x1b[1A\rThread 4 encoding file $(i-start+1) of $(stop-start+1)...\x1b[1B") : nothing
                
                tensor = encode(context, encoder_mode; fragment_size = fragment_size, fragment_groups = fragment_groups, verbose = false)

                if local_tensor === nothing
                    local_tensor = tensor
                else
                    local_tensor = merge_complete_tensors(local_tensor.header, 
                        local_tensor.forward_markov, tensor.forward_markov,
                        local_tensor.vocabulary.id_to_token, tensor.vocabulary.id_to_token)
                end
            end

            put!(result_channel, local_tensor)
        end

        return fetch(final_task)
    end

    """
        function merge_ctensors(tensorsfile1, tensorsfile2; merge_mult = 1)
    
    Merges the weights from tensordicts.

    ## Arguments
    - `tensorsfile1`: Tensordict the first.
    - `tensorsfile2`: Tensordict the second.
    - `merge_mult`: Tensordict the second.
    """
    function merge_ctensors(
        tensors1,
        tensors2;
        merge_mult = 1
    )
        if (tensors1.header.encoding_method != tensors2.header.encoding_method) || (tensors1.header.encoding_method == "unknown")
            error("Headers do not match (got '$(tensors1.header.encoding_method)' and '$(tensors2.header.encoding_method)').")
        end

        # The two models hold independent vocabularies, so a given integer id in
        # one does not denote the same token in the other. merge_complete_tensors
        # rebuilds a combined vocabulary and remaps both chains before summing;
        # naively merging the integer-keyed dicts would conflate unrelated tokens.

        # Scale the second model's transition counts so it contributes at the
        # requested ratio relative to the first (the README's weighting use-case).
        scaled_markov2 = tensors2.forward_markov
        if merge_mult != 1
            scaled_markov2 = Dict{Int, Dict{Int, Float64}}()
            for (id, transitions) in tensors2.forward_markov
                scaled_markov2[id] = Dict{Int, Float64}(next_id => weight * merge_mult for (next_id, weight) in transitions)
            end
        end

        return merge_complete_tensors(
            tensors1.header,
            tensors1.forward_markov,
            scaled_markov2,
            tensors1.vocabulary.id_to_token,
            tensors2.vocabulary.id_to_token
        )
    end
end