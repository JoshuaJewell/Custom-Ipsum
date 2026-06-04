module ModelIO
    using ..Types

    export save_model, load_model

    const MAGIC = b"CIPM"
    const FORMAT_VERSION = UInt8(2)
    const FLAG_INTEGRAL_WEIGHTS = UInt8(0x01)

    ## Unsigned LEB128 varints
    # Seven payload bits per byte, least significant group first.
    function write_uvarint(io::IO, value::Integer)
        value >= 0 || throw(ArgumentError("write_uvarint: value must be non-negative, got $value"))
        v = UInt64(value)
        while true
            byte = UInt8(v & 0x7f)
            v >>= 7
            if v != 0
                write(io, byte | 0x80)
            else
                write(io, byte)
                break
            end
        end
        return nothing
    end

    function read_uvarint(io::IO)
        result = UInt64(0)
        shift = 0
        while true
            byte = read(io, UInt8)
            result |= UInt64(byte & 0x7f) << shift
            (byte & 0x80) == 0 && break
            shift += 7
            shift >= 70 && throw(ArgumentError("malformed varint: exceeds 64-bit range"))
        end
        return result
    end

    ## Length-prefixed UTF-8 strings
    function write_string(io::IO, s::AbstractString)
        bytes = codeunits(s)
        write_uvarint(io, length(bytes))
        write(io, bytes)
        return nothing
    end

    function read_string(io::IO)
        n = Int(read_uvarint(io))
        bytes = read(io, n)
        length(bytes) == n || throw(EOFError())
        return String(bytes)
    end

    # True when every transition weight, across the outer and all interior
    # tables, equals its rounding.
    function all_weights_integral(tensors)
        ok(fm) = begin
            for inner in values(fm), w in values(inner)
                # Only strictly-positive integers are safe for the varint path;
                # anything else (fractional, zero, negative) falls back to Float64.
                (w > 0 && w == round(w)) || return false
            end
            return true
        end
        ok(tensors.forward_markov) || return false
        for tbl in values(tensors.interiors)
            ok(tbl) || return false
        end
        return true
    end

    # Write one Markov table: state count, then per state its id, its transition
    # count, and each (next id, weight) pair.
    function write_markov(io::IO, fm, integral::Bool)
        write_uvarint(io, length(fm))
        for (state_id, transitions) in fm
            write_uvarint(io, state_id)
            write_uvarint(io, length(transitions))
            for (next_id, weight) in transitions
                write_uvarint(io, next_id)
                integral ? write_uvarint(io, UInt64(weight)) :
                           write(io, htol(reinterpret(UInt64, weight)))
            end
        end
        return nothing
    end

    function read_markov(io::IO, integral::Bool)
        state_count = Int(read_uvarint(io))
        fm = Dict{Int, Dict{Int, Float64}}()
        sizehint!(fm, state_count)
        for _ in 1:state_count
            state_id = Int(read_uvarint(io))
            t = Int(read_uvarint(io))
            inner = Dict{Int, Float64}()
            sizehint!(inner, t)
            for _ in 1:t
                next_id = Int(read_uvarint(io))
                weight = integral ? Float64(read_uvarint(io)) :
                         reinterpret(Float64, ltoh(read(io, UInt64)))
                inner[next_id] = weight
            end
            fm[state_id] = inner
        end
        return fm
    end

    """
        save_model(path, tensors)

    Write a `CompleteTensors` to `path` in the CIPM v2 format. Integral weights are
    stored as varints; if any weight is fractional, all weights fall back to
    little-endian Float64. `token_to_id` is not written; it is rebuilt on load.
    """
    function save_model(path::AbstractString, tensors::CompleteTensors)
        integral = all_weights_integral(tensors)
        # Assemble the whole file in memory, then write it in one pass: the inner
        # loops emit many single-byte varints, far cheaper against an IOBuffer
        # than against a raw file handle.
        io = IOBuffer()
        write(io, MAGIC)
        write(io, FORMAT_VERSION)
        write(io, integral ? FLAG_INTEGRAL_WEIGHTS : UInt8(0))

        write_string(io, tensors.header.encoding_method)
        write_string(io, tensors.header.metadata)

        id_to_token = tensors.vocabulary.id_to_token
        write_uvarint(io, length(id_to_token))
        for token in id_to_token
            write_string(io, token)
        end

        write_markov(io, tensors.forward_markov, integral)

        # Interior tables (v2): a class count, then each class name and its table.
        write_uvarint(io, length(tensors.interiors))
        for (cls, tbl) in tensors.interiors
            write_string(io, String(cls))
            write_markov(io, tbl, integral)
        end

        write(path, take!(io))
        return nothing
    end

    """
        load_model(path) -> CompleteTensors

    Read a CIPM v1 or v2 model written by `save_model`. A v1 model loads with
    empty interior tables. Throws `ArgumentError` on a bad magic signature or an
    unsupported version.
    """
    function load_model(path::AbstractString)
        io = IOBuffer(read(path))

        magic = read(io, length(MAGIC))
        magic == MAGIC || throw(ArgumentError(
            "Not a Custom Ipsum model file (bad magic). Pre-existing Serialization " *
            "models are unsupported; regenerate them from source."))
        version = read(io, UInt8)
        version == 1 || version == 2 || throw(ArgumentError(
            "Unsupported model format version $version; this build reads versions 1 and 2."))
        flags = read(io, UInt8)
        integral = (flags & FLAG_INTEGRAL_WEIGHTS) != 0

        encoding_method = read_string(io)
        metadata = read_string(io)

        vocab_n = Int(read_uvarint(io))
        id_to_token = Vector{String}(undef, vocab_n)
        for i in 1:vocab_n
            id_to_token[i] = read_string(io)
        end
        token_to_id = Dict{String,Int}(token => i for (i, token) in enumerate(id_to_token))

        forward_markov = read_markov(io, integral)

        interiors = Dict{Symbol, Dict{Int, Dict{Int, Float64}}}()
        if version >= 2
            class_count = Int(read_uvarint(io))
            for _ in 1:class_count
                cls = Symbol(read_string(io))
                interiors[cls] = read_markov(io, integral)
            end
        end

        return CompleteTensors(
            Header(encoding_method, metadata),
            Vocabulary(token_to_id, id_to_token),
            forward_markov,
            interiors
        )
    end
end
