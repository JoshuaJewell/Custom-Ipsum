## Test suite for Custom Ipsum
# Run with: julia --project=. test/runtests.jl   (or via Pkg.test)
using CustomIpsum
using Test, Random

# A small but repetitive corpus so the Markov chain has genuine branching.
const CTX = "the cat sat on the mat. the dog sat on the log. a cat met a dog by the mat."

# Sum every transition count in a model; used to check merge arithmetic.
totalweight(t) = sum(sum(values(inner)) for inner in values(t.forward_markov))

@testset "Custom Ipsum" begin
    tensors = encode(CTX, "sanger", fragment_size = 4, fragment_groups = 2, verbose = false)

    @testset "sanger baseline still works" begin
        Random.seed!(1)
        out = decode(tensors, max_tokens = 32, stream = false, temperature = 1)
        @test out isa AbstractString
        @test !isempty(out)
    end

    @testset "streaming decode matches non-streaming" begin
        # Same seed, same sampled path: the streamed characters, concatenated,
        # must equal what the non-streaming path returns.
        Random.seed!(7)
        expected = CustomIpsum.Decoder.sanger_decoder(tensors, 20, false, 0, false, 1)

        # redirect_stdout needs a file or pipe, so capture through a temp file.
        Random.seed!(7)
        got = mktemp() do path, io
            redirect_stdout(() -> CustomIpsum.Decoder.sanger_decoder(tensors, 20, true, 0, false, 1), io)
            flush(io)
            read(path, String)
        end

        @test got == expected
    end

    @testset "streaming with show_tokens does not throw" begin
        @test redirect_stdout(devnull) do
            Random.seed!(7)
            CustomIpsum.Decoder.sanger_decoder(tensors, 10, true, 0, true, 1)
            true
        end
    end

    @testset "merge_ctensors sums transition counts" begin
        merged = merge_ctensors(tensors, tensors)
        @test totalweight(merged) ≈ 2 * totalweight(tensors)
    end

    @testset "merge_ctensors honours merge_mult" begin
        merged = merge_ctensors(tensors, tensors; merge_mult = 0.5)
        @test totalweight(merged) ≈ 1.5 * totalweight(tensors)
    end

    @testset "merge_ctensors rejects mismatched headers" begin
        HeaderT = typeof(tensors.header)
        TensorsT = typeof(tensors)
        other = TensorsT(HeaderT("default", tensors.header.metadata),
                         tensors.vocabulary, tensors.forward_markov)
        @test_throws ErrorException merge_ctensors(tensors, other)
    end

    @testset "legacy encoder modes fail cleanly" begin
        @test_throws ErrorException encode(CTX, "default", verbose = false)
    end

    @testset "legacy decoder modes fail cleanly" begin
        HeaderT = typeof(tensors.header)
        TensorsT = typeof(tensors)
        legacy = TensorsT(HeaderT("beamsearch", tensors.header.metadata),
                          tensors.vocabulary, tensors.forward_markov)
        @test_throws ErrorException decode(legacy, max_tokens = 8)
    end

    @testset "model round-trips through save/load" begin
        path = tempname() * ".ctensors"
        save_model(path, tensors)
        loaded = load_model(path)
        rm(path; force = true)
        @test loaded.header.encoding_method == tensors.header.encoding_method
        @test loaded.header.metadata == tensors.header.metadata
        @test loaded.vocabulary.id_to_token == tensors.vocabulary.id_to_token
        @test loaded.vocabulary.token_to_id == tensors.vocabulary.token_to_id
        @test loaded.forward_markov == tensors.forward_markov
        @test loaded.interiors == tensors.interiors
    end

    @testset "decode is reproducible after a round-trip" begin
        # A single-transition chain so generation cannot depend on Dict order.
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        id_to_token = ["<BOS>", "alpha", "beta", "gamma"]
        token_to_id = Dict(t => i for (i, t) in enumerate(id_to_token))
        fm = Dict(1 => Dict(2 => 3.0), 2 => Dict(3 => 1.0), 3 => Dict(4 => 1.0))
        det = C(H("sanger", "test"), V(token_to_id, id_to_token), fm)

        path = tempname() * ".ctensors"
        save_model(path, det)
        loaded = load_model(path)
        rm(path; force = true)

        Random.seed!(1)
        before = decode(det, max_tokens = 8, stream = false)
        Random.seed!(1)
        after = decode(loaded, max_tokens = 8, stream = false)
        @test before == after
        @test after == "alphabetagamma"
    end

    @testset "fractional weights survive a round-trip" begin
        merged = merge_ctensors(tensors, tensors; merge_mult = 0.5)
        path = tempname() * ".ctensors"
        save_model(path, merged)
        loaded = load_model(path)
        rm(path; force = true)
        @test loaded.forward_markov == merged.forward_markov
        @test totalweight(loaded) ≈ totalweight(merged)
    end

    @testset "load_model rejects a non-model file" begin
        path = tempname() * ".ctensors"
        write(path, "this is definitely not a Custom Ipsum model")
        @test_throws ArgumentError load_model(path)
        rm(path; force = true)
    end

    @testset "load_model rejects an unsupported version" begin
        path = tempname() * ".ctensors"
        open(path, "w") do io
            write(io, b"CIPM")
            write(io, UInt8(99))
        end
        @test_throws ArgumentError load_model(path)
        rm(path; force = true)
    end

    @testset "load_model surfaces truncation as a read error" begin
        # A valid preamble (magic, version, flags) but no body: reading the first
        # header string hits end-of-file.
        path = tempname() * ".ctensors"
        open(path, "w") do io
            write(io, b"CIPM")
            write(io, UInt8(1))
            write(io, UInt8(0))
        end
        @test_throws EOFError load_model(path)
        rm(path; force = true)
    end

    @testset "an empty model round-trips" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        empty = C(H("sanger", ""), V(Dict{String,Int}(), String[]), Dict{Int,Dict{Int,Float64}}())
        path = tempname() * ".ctensors"
        save_model(path, empty)
        loaded = load_model(path)
        rm(path; force = true)
        @test loaded.header.metadata == ""
        @test loaded.vocabulary.id_to_token == String[]
        @test loaded.forward_markov == Dict{Int,Dict{Int,Float64}}()
    end

    @testset "dialogue boundary bridge" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors

        # Model A opens with "hi\n" (a clean message end), whose next-start token
        # is "yo". Model B is known to begin messages with "yo", so the baton must
        # pass to B at that shared token.
        aTok = ["<BOS>", "hi\n", "yo"]
        aFm = Dict(1 => Dict(2 => 1.0), 2 => Dict(3 => 1.0))
        A = C(H("sanger", "a"), V(Dict(t => i for (i, t) in enumerate(aTok)), aTok), aFm)

        bTok = ["<BOS>", "yo", "there\n"]
        bFm = Dict(1 => Dict(2 => 2.0), 2 => Dict(3 => 1.0))
        B = C(H("sanger", "b"), V(Dict(t => i for (i, t) in enumerate(bTok)), bTok), bFm)

        @test beginnings(B) == Dict("yo" => 2.0)
        @test haskey(beginnings(A), "hi\n")

        Random.seed!(1)
        turns = dialogue(A, B; max_bubbles = 4)
        @test length(turns) == 4
        @test all(t -> !isempty(strip(t[2])), turns)   # no hollow bubbles
        @test all(t -> t in [(1, "hi"), (2, "yothere")], turns)
        @test any(t -> t[1] == 1, turns)               # A appears
        @test any(t -> t[1] == 2, turns)               # B appears via the seam
    end

    @testset "the walk splits embedded newlines and drops empties" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors

        # A single token carrying two newlines must split into two bubbles, with
        # the trailing empty segment dropped. We probe the walk directly, since the
        # public dialogue then selects a window out of a long pool.
        tok = ["<BOS>", "x\ny\n"]
        fm = Dict(1 => Dict(2 => 1.0))
        M = C(H("sanger", "m"), V(Dict(t => i for (i, t) in enumerate(tok)), tok), fm)

        Random.seed!(1)
        pool = CustomIpsum.Dialogue.walk(M, M, 4, 1, 3)
        @test all(t -> !isempty(strip(t[2])), pool)
        @test all(t -> t[2] in ["x", "y"], pool)
        @test (1, "x") in pool
        @test (1, "y") in pool
    end

    @testset "dialogue alternates by forced handoff when no seam is shared" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors

        # A and B share no beginning, so no seam can form; the walk must still
        # terminate and must hand off (forced) so both speakers appear.
        aTok = ["<BOS>", "aaa\n"]
        aFm = Dict(1 => Dict(2 => 1.0))
        A = C(H("sanger", "a"), V(Dict(t => i for (i, t) in enumerate(aTok)), aTok), aFm)

        bTok = ["<BOS>", "bbb\n"]
        bFm = Dict(1 => Dict(2 => 1.0))
        B = C(H("sanger", "b"), V(Dict(t => i for (i, t) in enumerate(bTok)), bTok), bFm)

        Random.seed!(1)
        turns = dialogue(A, B; max_bubbles = 4, max_run = 1)
        @test length(turns) <= 4
        @test !isempty(turns)
        @test any(t -> t[1] == 1, turns)
        @test any(t -> t[1] == 2, turns)
        @test all(t -> !isempty(strip(t[2])), turns)
    end

    @testset "volume sums a model's transition mass" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        tok = ["<BOS>", "x", "y"]
        fm = Dict(1 => Dict(2 => 3.0), 2 => Dict(3 => 1.5))
        M = C(H("sanger", "v"), V(Dict(t => i for (i, t) in enumerate(tok)), tok), fm)
        @test CustomIpsum.Dialogue.volume(M) == 4.5
    end

    @testset "pick_window favours the volume ratio then liveliness" begin
        pw = CustomIpsum.Dialogue.pick_window

        # An even target should land on the balanced, alternating tail rather than
        # the all-speaker-one head.
        even = [(1, "a"), (1, "b"), (1, "c"), (1, "d"), (2, "e"), (1, "f"), (2, "g"), (1, "h")]
        win = pw(even, 4, 0.5)
        @test length(win) == 4
        @test count(b -> b[1] == 1, win) == 2
        @test win == [(1, "d"), (2, "e"), (1, "f"), (2, "g")]

        # A three-to-one target should pick a window where speaker one dominates.
        skew = [(2, "a"), (2, "b"), (1, "c"), (2, "d"), (1, "e"), (1, "f"), (1, "g"), (2, "h")]
        @test count(b -> b[1] == 1, pw(skew, 4, 0.75)) == 3
    end

    @testset "CompleteTensors interiors field defaults empty" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        tok = ["<BOS>", "x"]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        fm = Dict(1 => Dict(2 => 1.0))

        # The three-argument construction the rest of the suite uses must keep
        # working and yield empty interiors.
        flat = C(H("sanger", ""), v, fm)
        @test flat.interiors == Dict{Symbol,Dict{Int,Dict{Int,Float64}}}()
        @test typeof(flat.interiors) == Dict{Symbol,Dict{Int,Dict{Int,Float64}}}

        # The four-argument construction carries interior tables.
        interiors = Dict(:paren => Dict(2 => Dict(2 => 1.0)))
        nested = C(H("sanger", ""), v, fm, interiors)
        @test nested.interiors[:paren][2][2] == 1.0
    end

    @testset "delimiter_event classifies delimiters" begin
        de = CustomIpsum.Utils.delimiter_event
        # Parentheses are directional regardless of neighbours.
        @test de('x', '(', 'y') == (:open, :paren)
        @test de('y', ')', 'z') == (:close, :paren)
        # A quote opens when its outer side is a boundary and inner side is not.
        @test de(' ', '"', 'd') == (:open, :quote)
        @test de('\0', '"', 'd') == (:open, :quote)
        # A bracket is a boundary too, so a quote may open right inside a paren.
        @test de('(', '"', 'd') == (:open, :quote)
        # ...and closes in the mirror case (quote then space or punctuation).
        @test de('d', '"', ' ') == (:close, :quote)
        @test de('d', '"', '.') == (:close, :quote)
        @test de('d', '"', '\0') == (:close, :quote)
        # Letters either side: the glyph reads as a contraction apostrophe, not a quote.
        @test de('n', '"', 't') == (:none, :none)
        # A quote bounded on both sides is ambiguous and counts as no delimiter.
        @test de(' ', '"', ' ') == (:none, :none)
        # An ordinary character is no delimiter.
        @test de('a', 'b', 'c') == (:none, :none)
    end

    @testset "scan_depth_classes tracks delimiter depth" begin
        sdc = CustomIpsum.Utils.scan_depth_classes
        active = Set([:paren, :quote])

        # "a (b) c": positions inside the parentheses are :paren, the rest :outer.
        ctx = "a (b) c"
        cls = sdc(ctx, active)
        @test cls[1] == :outer            # 'a'
        @test cls[3] == :paren            # '(' just opened
        @test cls[4] == :paren            # 'b'
        @test cls[5] == :outer            # ')' just closed
        @test cls[7] == :outer            # 'c'

        # A quote nested inside parentheses sits at :quote.
        ctx2 = "x (\"y\") z"
        cls2 = sdc(ctx2, active)
        @test cls2[4] == :quote           # the opening '"' itself sits inside the parens
        @test cls2[5] == :quote           # 'y' is inside the quote inside parens

        # With no active classes every position is :outer (the flat model).
        @test all(==(:outer), sdc(ctx, Set{Symbol}()))
    end

    @testset "sanger_split returns aligned source spans" begin
        ctx = "abcdef"
        tokens, spans = CustomIpsum.Utils.sanger_split(ctx, 3, 1)
        chars = collect(ctx)
        @test length(tokens) == length(spans)
        # The leading synthetic separator carries the sentinel span.
        @test tokens[1] == "\n"
        @test spans[1] == (0, 0)
        # Every real token equals the substring named by its span.
        for (tok, (a, b)) in zip(tokens, spans)
            (a, b) == (0, 0) && continue
            @test tok == join(chars[a:b])
        end

        # Two groups exercise the alternating-size path too, whose spans must
        # likewise name the exact substring they were cut from.
        toks2, sps2 = CustomIpsum.Utils.sanger_split("abcdefgh", 3, 2)
        chars2 = collect("abcdefgh")
        @test length(toks2) == length(sps2)
        for (tok, (a, b)) in zip(toks2, sps2)
            (a, b) == (0, 0) && continue
            @test tok == join(chars2[a:b])
        end
    end

    @testset "interior tables partition the flat chain" begin
        # Sum a model's outer and interior edges into one (src,dst) => weight map.
        function unioned(t)
            acc = Dict{Tuple{Int,Int},Float64}()
            add(fm) = for (a, inner) in fm, (b, w) in inner
                acc[(a, b)] = get(acc, (a, b), 0.0) + w
            end
            add(t.forward_markov)
            for tbl in values(t.interiors); add(tbl); end
            return acc
        end

        ctx = "say (an aside) and \"a quote\" then stop."
        flat = encode(ctx, "sanger", fragment_size = 3, fragment_groups = 1,
                      delimiters = Symbol[], verbose = false)
        routed = encode(ctx, "sanger", fragment_size = 3, fragment_groups = 1,
                        verbose = false)

        # delimiters = [] reproduces the flat model: no interiors at all.
        @test isempty(flat.interiors)
        # Default-on routing actually fills interior tables.
        @test !isempty(routed.interiors)
        # The two share a vocabulary (same tokens, same ids), so the unions are
        # directly comparable, and routing only re-files edges, never invents or
        # drops them.
        @test routed.vocabulary.id_to_token == flat.vocabulary.id_to_token
        @test unioned(routed) == unioned(flat)
    end

    @testset "interiors round-trip through save/load (v2)" begin
        ctx = "say (an aside) and \"a quote\" then stop."
        routed = encode(ctx, "sanger", fragment_size = 3, fragment_groups = 1,
                        verbose = false)
        @test !isempty(routed.interiors)

        path = tempname() * ".ctensors"
        save_model(path, routed)
        loaded = load_model(path)
        rm(path; force = true)

        @test loaded.forward_markov == routed.forward_markov
        @test loaded.interiors == routed.interiors
    end

    @testset "a v1 file loads with empty interiors" begin
        # Hand-write a minimal version-1 model (no interiors block) using the
        # module's own writers, then confirm it loads and reads as flat.
        M = CustomIpsum.ModelIO
        path = tempname() * ".ctensors"
        open(path, "w") do io
            write(io, b"CIPM")
            write(io, UInt8(1))          # version 1
            write(io, UInt8(0x01))       # integral-weights flag
            M.write_string(io, "sanger")
            M.write_string(io, "")       # metadata
            M.write_uvarint(io, 2)       # vocabulary: <BOS>, "x"
            M.write_string(io, "<BOS>")
            M.write_string(io, "x")
            M.write_uvarint(io, 1)       # one outer state
            M.write_uvarint(io, 1)       # state id 1
            M.write_uvarint(io, 1)       # one transition
            M.write_uvarint(io, 2)       # to id 2
            M.write_uvarint(io, 1)       # weight 1
        end
        loaded = load_model(path)
        rm(path; force = true)
        @test loaded.vocabulary.id_to_token == ["<BOS>", "x"]
        @test loaded.forward_markov == Dict(1 => Dict(2 => 1.0))
        @test isempty(loaded.interiors)
    end

    @testset "nested decode closes a parenthetical naturally" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        # BOS -> "a (" (outer) -> "b)" (inside parens) -> " c" (back outside).
        tok = ["<BOS>", "a (", "b)", " c"]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        outer = Dict(1 => Dict(2 => 1.0), 3 => Dict(4 => 1.0))
        interiors = Dict(:paren => Dict(2 => Dict(3 => 1.0)))
        m = C(H("sanger", ""), v, outer, interiors)

        Random.seed!(1)
        out = decode(m, max_tokens = 8, stream = false)
        @test out == "a (b) c"
    end

    @testset "nested decode synthesises a close on an interior dead-end" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        # Inside the parens we reach "x", which has no interior successor and does
        # not itself carry ')': the walk must close the bracket and stop.
        tok = ["<BOS>", "(", "x"]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        outer = Dict(1 => Dict(2 => 1.0))
        interiors = Dict(:paren => Dict(2 => Dict(3 => 1.0)))
        m = C(H("sanger", ""), v, outer, interiors)

        Random.seed!(1)
        out = decode(m, max_tokens = 8, stream = false)
        @test out == "(x)"
    end

    @testset "flat decode is unchanged when interiors are empty" begin
        # The deterministic chain from the round-trip test still yields its exact
        # string, proving the flat path is untouched by the dispatch.
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        tok = ["<BOS>", "alpha", "beta", "gamma"]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        fm = Dict(1 => Dict(2 => 3.0), 2 => Dict(3 => 1.0), 3 => Dict(4 => 1.0))
        m = C(H("sanger", "test"), v, fm)
        Random.seed!(1)
        @test decode(m, max_tokens = 8, stream = false) == "alphabetagamma"
    end

    @testset "a token-final closing quote at depth zero does not reopen" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        # The token "hi\"" ends in a quote preceded by a letter: a close, not an
        # open. At depth zero it must be left alone, not pushed as a fresh quote
        # (which would strand the walk in the :quote table and emit a stray '"').
        tok = ["<BOS>", "hi\"", " ok"]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        outer = Dict(1 => Dict(2 => 1.0), 2 => Dict(3 => 1.0))
        interiors = Dict(:quote => Dict(3 => Dict(3 => 1.0)))
        m = C(H("sanger", ""), v, outer, interiors)

        Random.seed!(1)
        @test decode(m, max_tokens = 8, stream = false) == "hi\" ok"
    end

    @testset "a token-final opening quote still opens the quote chain" begin
        H = CustomIpsum.Types.Header
        V = CustomIpsum.Types.Vocabulary
        C = CustomIpsum.Types.CompleteTensors
        # "said \"" ends in a quote preceded by a space: a genuine open. The walk
        # must enter the :quote chain, emit "hi\"" from it, then close on that
        # token's own trailing quote.
        tok = ["<BOS>", "said \"", "hi\""]
        v = V(Dict(t => i for (i, t) in enumerate(tok)), tok)
        outer = Dict(1 => Dict(2 => 1.0))
        interiors = Dict(:quote => Dict(2 => Dict(3 => 1.0)))
        m = C(H("sanger", ""), v, outer, interiors)

        Random.seed!(1)
        @test decode(m, max_tokens = 8, stream = false) == "said \"hi\""
    end
end
