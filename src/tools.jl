module Tools
    include("types.jl")
    include("encoder.jl")
    include("decoder.jl")
    include("utils.jl")

    using .Types, .Encoder, .Decoder, Distributed, .Utils

    export encode_multiple, encoder_decoder, merge_ctensors

    function encoder_decoder(
        context,
        mode = "default";
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
        function encode_multiple(path_to_context = "../data/contexts/", context_filename = "context", context_file_no; mode = "equal")
    
    Encodes multiple contexts from a set of files and merges the weights. Store contexts in format:
        context1.txt
        context2.txt
        etc...

    ## Arguments
    - `path_to_context` (default: "../data/contexts/"): Where your context files to encode and merge are.
    - `context_filename` (default: "context"): The unchanged part of your contexts' filenames.
    - `context_file_no` (default: 2): The number of context files, indexed from 1.

    ## Keyword Arguments
    - `merge_mult` (optional, default: 1): How much weight to give added tensordict. (WIP)
    - `encoder_mode` (optional, default: "default"): The encoder mode. Can be "default", or "sanger".
    - `fragment_size` (optional, default: 1): How long (in characters) for tokens to be. Attempts to find optimal when set to 1. Only relevant if `mode` is "sanger".
    - `fragment_groups` (optional, default: 1): How many different fragment sizes should be parsed (high values not recommended). Only relevant if `mode` is "sanger" and `fragment_size` is specified.

    """
    function encode_multiple(
        path_to_context = "../data/contexts/",
        context_filename = "context",
        context_file_no = 2;
        merge_mult = 1,
        encoder_mode = "sanger",
        exclude = [""],
        fragment_size = 1,
        fragment_groups = 1
    )
        encoder_mode = lowercase(encoder_mode)

        contexts = []
        for i in 1:context_file_no
            context = read("$path_to_context$context_filename$i.txt", String)
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
                        final_merged_tensors = merge_tensordicts(final_merged_tensors, result)
                    end
                end
                finished_count += 1
            end

            tensors = pack_ctensors(encoder_mode, args, final_merged_tensors)

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
                if encoder_mode == "sanger"
                    tensor = sanger_encoder(context, fragment_size, fragment_groups, exclude, false)
                    args = "Fragmentation: $fragment_size by $fragment_groups. $exclude excluded."
                else
                    tensor = default_encoder(context, verbose=false)
                    args = "Sentence enders: $end_punctuation; Preserved tokens: $preserve_tokens. $exclude excluded."
                end

                if local_tensor === nothing
                    local_tensor = tensor
                else
                    local_tensor = merge_tensordicts(local_tensor, tensor)
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
        tensorsfile1,
        tensorsfile2;
        merge_mult = 1
    )
        if (tensorsfile1.header != tensorsfile2.header) || (tensorsfile1.header.encoding_method == "unkown")
            error("Headers do not match")
        else
            print("Headers match, proceeding to merge.")
        end

        tensors1 = unpack_ctensors(tensorsfile1) # (encoding_method, metadata, forward_markov, reverse_markov, token_index)
        tensors2 = unpack_ctensors(tensorsfile2) #          1            2            3               4             5

        # Merge markov chains
        merged_forward_markov = merge_tensordicts(
            tensors1[3],
            tensors2[3];
            merge_mult = merge_mult
        )
        merged_reverse_markov = merge_tensordicts(
            tensors1[4],
            tensors2[4];
            merge_mult = merge_mult
        )

        # Concatenate token_index (not using it yet anyway sooo...)
        merged_token_index = vcat(tensors1[5], tensors2[5])

        # Create new CompleteTensors instance
        merged_tensor = pack_ctensors(tensors1[1], tensors1[2], merged_forward_markov, merged_reverse_markov, merged_token_index)

        return merged_tensor
    end
end