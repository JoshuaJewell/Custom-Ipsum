<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JoshuaJewell.Custom-Ipsum&"  />
</div>

# Custom Ipsum

Placeholder text generator using markov chains. Its original incarnation attempted to create realistic prose given many grammar rules, while `sanger` mode, which has superseded it, and is now the only one supported, pretty much assumes nothing about anything, generation included, and works over raw character fragments instead of words. This makes it completely language/alphabet-agnostic. There are some fairly cool principles in here, and I would hazard a guess that this is the most advanced program of its type (as long as you maintain a very narrow view of what NLP can mean - e.g. LLMs don't count).

## Background

`sanger` mode takes its name, and its shape, from Sanger sequencing and de novo assembly. A sequencing read is a short fragment of a much longer molecule, and an assembler has to rebuild the whole thing from many overlapping fragments with no reference to check itself against. So the corpus here is cut into short overlapping fragments at every offset and length the encoder asks for, ignoring word boundaries entirely, and generation is a probabilistic walk across them: at each step the current fragment votes, by observed frequency, for what follows it, and the walk samples from that.

One of the inspirations for this was BleachBit's chaff generator, which produces decoy documents from source material (Hillary Clinton's emails and 2600 magazine) so that secure deletion has something plausible to hide behind. `sanger` mode's fragment-and-reassemble approach, and its delimiter handling below, both came out of wanting a generator that could stand in that tradition while doing something slightly more considered than shuffling words.

## Features

### Sanger encoding: fragment-based Markov generation

The corpus is tokenised into overlapping character fragments at every valid offset within the chosen `fragment_size`. Left at its default, fragment length is picked automatically from the corpus's own average word length.

`fragment_groups` trains several fragment lengths into the same model at once, both independently and in alternating pairs (I was very proud of this method), so the walk has more than one order of context to hand as it proceeds.

### Delimiter-aware generation

Sanger encoding routes matched delimiters through their own interior chains, so that brackets and quotes open and close coherently. A plain first-order Markov chain is a finite-state machine and cannot keep delimiters balanced, but adding a stack allows it to. Parentheses `()` and double quotes `""` are handled by default. Pass `delimiters = []` to `encode()` for standard flat behaviour, or a subset such as `delimiters = [:paren]` to route only some classes; a corpus with no delimiters is unaffected. Models are saved in the CIPM v2 format, which carries the interior chains, and v1 models still load and decode as flat.

### Two-voice dialogue

`dialogue(modelA, modelB)` stages a synthetic exchange between two independently trained models instead of one stream of text. Each speaker generates in its own chain. The baton passes at a clean message boundary whose opening fragment the other speaker is also known to begin with, or after a capped run of messages if no such shared opening turns up. The walk over-generates a long pool of turns and then picks the contiguous window whose speaker balance best matches the two models' relative text volume.

### Model merging and batch encoding

`merge_ctensors()` combines two saved models' transition weights into one, rebuilding a shared vocabulary, with an optional `merge_mult` to weight the second model's contribution against the first. `encode_multiple()` encodes every file in a directory of context files in parallel across threads and folds the results into a single merged model.

### Portable model format (CIPM)

Trained models are saved as `.ctensors` files in a small custom binary format: a version and flags byte, the vocabulary, the outer transition table, and, in v2, the interior delimiter tables. Transition weights are packed as varints when every weight happens to be a positive integer, and as raw floats otherwise. There is no external package dependency for any of this. Encoding, decoding, dialogue and model I/O all run on Julia's standard library (`Random`, `Distributed`, `Serialization`), so cloning the repository and having Julia 1.6 or later installed is pretty much it.

### Sampling controls

`decode()` takes a `temperature` (higher flattens the distribution toward diversity, lower sharpens it toward the most likely continuation...nope, I can't really see the difference either, but obviously the easiest steal from LLM sampling I could do), can stream output token by token at a configurable rate, and can annotate each streamed token with the probability it was sampled at (`show_tokens`), which doubles as a rough per-token confidence score.

## Usage

```
./run.sh [context_or_model] [max_tokens]
```

Both arguments are optional and default to `data/contexts/macbeth.txt` and 512 tokens. The first may be a plain-text context file or a previously saved `.ctensors` model. Given a text context, the script encodes it with `fragment_size = 5, fragment_groups = 3` and saves the result to `data/models/local-<name>.ctensors` before generating from it.

Further generation customisation will require editing `TextGen.jl`, but I assure you my functions are intuitive and the docstrings are hopefully clear.

Run the test suite with:

```
julia --project=. test/runtests.jl
```

## Examples

### `macbeth.ctensors` (8.8 MB trained model):

```
  $ ./run.sh data/models/local-macbeth.ctensors 200

  LADY MACDUFF.
  Poor prattler, how thou talk'st!

  Enter a Messenger.

  Thou com'st to use thy tongue; thy story quickly.

  MESSENGER.
  Gracious my lord,
  I should report that which I say I saw,
  But know not how to do't.

  MACBETH.
  Your children shall be kings,
  When mine are blanch'd with fear. What's the boy Malcolm?
  Was he not born of woman? The spirits that knits up the ravell'd sleave of care,
  The death of each day's life, sore labour's bath,
  Balm of hurt minds, great nature's second course,
  Chief nourisher in life's feast.
```

### `brackets.ctensors` (synthetic delimeter demo model):

```
  $ ./run.sh data/models/local-brackets.ctensors 200

  The witch (who was old) spoke softly. "Beware," she said, "the ides." A man
  (a strangers do) without a word. "Farewell," he called, "and good luck." The
  road (long and dark) wound on. She lit a candle (the last one) and waited.
  "Who goes there?" cried the guard. A shape (tall and silent) crossed the
  yard. "Peace," it whispered, "I mean no harm."...
```

## Potential applications

The engine underneath the placeholder text is a fragment-based Markov chain with optional multi-order blending and a pushdown extension for matched delimiters, and it has uses well outside generating ipsum.

- **Password and passphrase research.** Character-level fragment Markov chains over leaked password corpora similarly to how hashcat's and John the Ripper's Markov modes model guessability. Fed a password list instead of prose, this engine could estimate guess-order likelihood, or generate honeywords (decoy credentials statistically indistinguishable from real ones).
- **Chaff and decoy document generation**, BleachBit-adjacent suspicious file generation.
- **Bioinformatics null models.** Fixed-order Markov models over nucleotide k-mers are a standard technique for background sequence generation that preserves local composition (GC content, dinucleotide bias), used to test whether an observed motif is statistically significant. At least you can definitely generate realistic-looking nucleotide sequences.
- **RNA secondary-structure shape generation.** Feed it dot-bracket notation (`(((...)))..`) with `delimiters = [:paren]` and the existing pushdown mechanism guarantees well-nested output, giving a generator of plausible fold topology: branch lengths, loop sizes, depth distribution.
- **Procedural music generation.** Markov-chain melody generation is apparently a thing, and a corpus of tunes in a plain-text notation like ABC is a potential fit here. The character-window split would want to respect note and duration tokens instead of cutting across them, and ABC's repeat markers `|:` `:|` are a matched-delimiter pair in exactly the shape `:paren` already handles, but will require generalisation of delimiter-aware delimitation.
- **Grammar-aware fuzzing.** Something that keeps delimiters balanced while still varying content statistically sits usefully between pure random fuzzing, which most parsers reject on sight for unbalanced input, and full grammar-based fuzzing, for probing parsers of nested formats.
- **Anomaly detection.** The per-step transition probability the decoder already computes during generation could be walked over a real held-out sequence instead of a generated one, giving a likelihood trace that flags where a sequence departs from the trained model. This principle can create a Markov-based anomaly detector for system-call or command logs.

## Moving forward

- **Order backoff.** `fragment_groups` currently merges edges from several fragment lengths into one flat table instead of trying the longest available context first and falling back when it is unseen. This would cut down on premature dead ends and on the near-verbatim quoting a single large fragment size produces on a small corpus. Every generation use case above benefits.
- **Anchored generation.** A way to resume a walk from a supplied prefix instead of only from `<BOS>`. Needed for keyword-seeded chaff, for code-completion-style use, and as the forward half of infilling.
- **A backward-trained companion chain.** Useful in its own right for capturing end-anchored regularities that a forward-only, start-seeded chain undersamples (common password suffixes, musical cadences), and, paired with anchored generation, for two-ended infilling: growing text forward from a left anchor and backward from a right anchor to meet in the middle.
- **Generalised delimiter classes.** Extending `delimiter_event` from single glyphs to short fixed strings or keyword pairs, past `(` `)` and `"`. This is what would let the model track ABC's `|:`/`:|` repeat markers, and separately Julia's `function`/`begin`/`if`/`for`/`module` paired with `end`, and any other programming language's grammar, which the current character-glyph mechanism cannot see at all.
- **A standalone scoring function**, exposing the decoder's per-step probability against externally supplied text and not only during generation. This sits underneath both quality filtering of generated output (keep only the most probable candidates) and the anomaly-detection case above.
- **A repetition guard**, against the short degenerate loops (A to B to A to B) that Markov generation falls into at low temperature. Cheap to add, and would visibly improve output quality across the board.

## Disclaimer

`data/contexts/fineweb-top5000.txt` was compiled using the top 5000 entries sorted by language score from the <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">🍷 FineWeb dataset</a> (Penedo _et al._ 2024). Any model generated from it, or from a merge that includes it, inherits its weights and its provenance. Therefore I assume no liability for any issues arising from such a model's outputs. The data may contain inappropriate content and are not intended for critical decision-making, and I advise against relying on them for such purposes.

## References

Penedo, G., Kydlíček, H., allal, L.B., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. (2024) 'The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale', available: https://doi.org/10.48550/ARXIV.2406.17557.
