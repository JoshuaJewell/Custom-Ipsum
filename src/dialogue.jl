## Two-voice dialogue over two sanger models. The walk stages a long transcript:
## each speaker generates messages in its own chain and, at a clean message seam,
## hands the baton to the other when its next beginning is one the other also uses
## (otherwise after a short run). We then select the contiguous window whose
## speaker mix best matches the pair's relative text volume and reads as the
## liveliest exchange. Bubbles are cut on newlines in the emitted character stream,
## so embedded newlines split correctly and empty segments are dropped.
module Dialogue
    using ..Types, ..Utils
    using Random

    export dialogue, beginnings, endings

    ## The message-start vocabulary: tokens that follow a message boundary,
    ## weighted by frequency, keyed by token string so two models can be matched.
    function beginnings(tensors)
        bos = tensors.vocabulary.token_to_id["<BOS>"]
        out = Dict{String,Float64}()
        haskey(tensors.forward_markov, bos) || return out
        for (id, w) in tensors.forward_markov[bos]
            tok = tensors.vocabulary.id_to_token[id]
            out[tok] = get(out, tok, 0.0) + w
        end
        return out
    end

    ## The message-end vocabulary: fragments terminating in a newline, weighted by
    ## their onward transitions. Exposed for inspection; the walk reads endings off
    ## each emitted token directly.
    function endings(tensors)
        out = Dict{String,Float64}()
        for id in 1:length(tensors.vocabulary.id_to_token)
            tok = tensors.vocabulary.id_to_token[id]
            if endswith(tok, "\n")
                w = haskey(tensors.forward_markov, id) ? sum(values(tensors.forward_markov[id])) : 0.0
                out[tok] = get(out, tok, 0.0) + w
            end
        end
        return out
    end

    ## Total transition mass, proportional to how much this speaker wrote; both
    ## models share encoding parameters, so the ratio reflects relative volume.
    function volume(tensors)
        total = 0.0
        for d in values(tensors.forward_markov)
            total += sum(values(d))
        end
        return total
    end

    ## Weighted pick over (key, weight) pairs, with the same temperature shaping as
    ## the default sampler.
    function weighted_pick(pairs, temperature)
        total = sum(w for (_, w) in pairs)
        total > 0 || return first(pairs)[1]
        probs = [w / total for (_, w) in pairs]
        probs = probs .^ temperature
        probs ./= sum(probs)
        r = rand()
        c = 0.0
        for (i, kv) in enumerate(pairs)
            c += probs[i]
            c >= r && return kv[1]
        end
        return last(pairs)[1]
    end

    ## Append a token's text to a speaker's in-progress line, cutting completed
    ## bubbles on every newline (embedded ones included) and dropping empty
    ## segments. Returns (line, clean_boundary, capped): clean_boundary is true when
    ## the token ended exactly at a newline, the only place a handoff may occur;
    ## capped is true once max_bubbles has been reached.
    function emit!(bubbles, speaker, line, token, max_bubbles)
        line *= token
        occursin('\n', line) || return (line, false, false)
        pieces = split(line, '\n')
        for seg in pieces[1:end-1]
            isempty(strip(seg)) && continue
            push!(bubbles, (speaker, String(strip(seg))))
            length(bubbles) >= max_bubbles && return (String(pieces[end]), true, true)
        end
        tail = String(pieces[end])
        return (tail, isempty(tail), false)
    end

    ## The boundary-bridge walk: generate up to max_bubbles staged turns.
    function walk(tA, tB, max_bubbles, temperature, max_run)
        models = (tA, tB)
        begins = (beginnings(tA), beginnings(tB))
        bos = (tA.vocabulary.token_to_id["<BOS>"], tB.vocabulary.token_to_id["<BOS>"])

        bubbles = Vector{Tuple{Int,String}}()
        cur = 1
        other = 2
        current_id = bos[cur]
        line = ""
        at_boundary = false
        run_msgs = 0
        guard = 0
        guard_max = max_bubbles * 400

        ## Flush any unfinished line as a final bubble for the given speaker.
        function flush_remnant!(speaker)
            if !isempty(strip(line))
                push!(bubbles, (speaker, String(strip(line))))
            end
            line = ""
        end

        ## Pass the baton: the other speaker opens afresh from its own beginnings.
        function force_handoff!()
            flush_remnant!(cur)
            cur, other = other, cur
            current_id = bos[cur]
            line = ""
            at_boundary = false
            run_msgs = 0
        end

        while length(bubbles) < max_bubbles && guard < guard_max
            guard += 1
            fwd = models[cur].forward_markov
            id2t = models[cur].vocabulary.id_to_token

            if at_boundary
                trans = get(fwd, current_id, Dict{Int,Float64}())
                seam = Tuple{Int,Float64}[]
                for (t, w) in trans
                    s = id2t[t]
                    haskey(begins[other], s) && push!(seam, (t, w * begins[other][s]))
                end

                if !isempty(seam) && run_msgs < max_run
                    ## Hand over at a shared beginning, which the other speaker emits.
                    s = id2t[weighted_pick(seam, temperature)]
                    flush_remnant!(cur)
                    length(bubbles) >= max_bubbles && break
                    cur, other = other, cur
                    current_id = models[cur].vocabulary.token_to_id[s]
                    run_msgs = 0
                    line, at_boundary, capped = emit!(bubbles, cur, "", s, max_bubbles)
                    capped && break
                    continue
                elseif run_msgs >= max_run || isempty(trans)
                    force_handoff!()
                    length(bubbles) >= max_bubbles && break
                    continue
                end
            elseif !haskey(fwd, current_id)
                force_handoff!()
                length(bubbles) >= max_bubbles && break
                continue
            end

            ## Generate the current speaker's next token.
            nxt = default_sampler(fwd, current_id, temperature)[1]
            current_id = nxt
            line, clean, capped = emit!(bubbles, cur, line, id2t[nxt], max_bubbles)
            capped && break
            at_boundary = clean
            clean && (run_msgs += 1)
        end

        if length(bubbles) < max_bubbles && !isempty(strip(line))
            push!(bubbles, (cur, String(strip(line))))
        end
        return bubbles
    end

    ## Choose the contiguous window of w bubbles whose speaker-1 fraction best
    ## matches the target pA, breaking ties toward livelier (more alternating)
    ## windows so a snippet that merely hits the ratio while one speaker dominates
    ## loses to one that genuinely goes back and forth.
    function pick_window(bubbles, w, pA)
        n = length(bubbles)
        n <= w && return bubbles
        best_i = 1
        best_score = -Inf
        for i in 1:(n - w + 1)
            ones = 0
            switches = 0
            for k in 0:(w - 1)
                bubbles[i + k][1] == 1 && (ones += 1)
                k < w - 1 && bubbles[i + k][1] != bubbles[i + k + 1][1] && (switches += 1)
            end
            a = ones / w
            alt = switches / (w - 1)
            score = -2.0 * abs(a - pA) + alt
            if score > best_score
                best_score = score
                best_i = i
            end
        end
        return bubbles[best_i:(best_i + w - 1)]
    end

    ## Stage a long exchange, then return the window that best fits the pair's
    ## relative volume and reads most like a conversation.
    function dialogue(tA, tB; max_bubbles = 8, temperature = 1, max_run = 3, pool = nothing)
        ## Over-generate about eight times the snippet (with a floor) so the window
        ## selection has a real field of alternation to choose from.
        poolN = pool === nothing ? max(64, max_bubbles * 8) : pool
        long = walk(tA, tB, poolN, temperature, max_run)
        length(long) <= max_bubbles && return long
        vA = volume(tA)
        vB = volume(tB)
        pA = (vA + vB) > 0 ? vA / (vA + vB) : 0.5
        return pick_window(long, max_bubbles, pA)
    end
end
