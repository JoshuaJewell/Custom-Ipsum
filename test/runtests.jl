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
end
