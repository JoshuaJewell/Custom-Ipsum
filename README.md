<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JoshuaJewell.Custom-Ipsum&"  />
</div>

# Custom Ipsum
Placeholder text generator using markov chains; one method assuming too much about the English language, the other assuming nothing about anything. Some fairly cool principles in here though, I would hazard a guess that this is the most advanced program of its type (as long as you maintain a very narrow view of what NLP can mean - c'mon LLM's don't count, do they?). 

## Delimiter-aware generation
Sanger encoding routes matched delimiters through their own interior chains, so
that brackets and quotes open and close coherently rather than being scattered by
a memoryless chain. A plain first-order Markov chain is a finite-state machine and
cannot keep delimiters balanced; adding a stack lifts it to a pushdown automaton,
which can. Parentheses `()` and double quotes `""` are handled by default. Pass
`delimiters = []` to `encode` for the older flat behaviour, or a subset such as
`delimiters = [:paren]` to route only some classes; a corpus with no delimiters is
unaffected. Models are saved in the CIPM v2 format, which carries the interior
chains, and v1 models still load and decode as flat.

## Disclaimer
`fineweb-top5000.tensordict` was generated using the top 5000 entries sorted by language score from the <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">🍷 FineWeb dataset</a> (Penedo _et al._ 2024). `samplemerged@7E-4.tensordict` also contains these weights at a ratio of 7.0x10-4 into my own data. Therefore, I assume no liability for any issues arising from these model's outputs. The data may contain inappropriate content and are not intended for critical decision-making, and I advise against relying on them for such purposes.

## References
Penedo, G., Kydlíček, H., allal, L.B., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. (2024) ‘The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale’, available: https://doi.org/10.48550/ARXIV.2406.17557.
