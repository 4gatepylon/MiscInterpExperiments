Can llama compress stuff? Apparently it's pretty good with known books but not so good with some random shit. With that said, to actually test this you would want to use more data at a time.

Did the most naive thing ever: just plain autoregression.
- Tried llama 8B with greedy decoding
- Algorithm is to just store indices that were wrong and a seed of the "initial text"
- 12:1 compression on gutenberg book at 2K tokens (maybe like 6KB... takes like 1min per KB unfort. since it's dont 100% sequentially); note we did not account for the model here
- 0.5 compression ratio on arbitrary reasoning dataset of prompts: near the theoretical worst case
- Also worst case when compressing sequences of those indices