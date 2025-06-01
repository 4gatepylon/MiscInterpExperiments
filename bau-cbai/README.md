# FYI
This is unfinished since I wasn't going to do the program, so it was on the margin better to spend my time elsewhere.

# High-level
This is my application for David Bau's CBAI stream (they asked for it to be in a Github repo, hope public is OK, since I'm submitting basically right before the deadline).

Below I describe my CoT and then I keep a running log of the experiments I make.

# Plan (CoT)
The task is outlined as follows:
```
Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.Type: fruit
List: [dog apple cherry bus cat grape bowl]
Answer: (
...
...
...


Your task:

1. create a dataset of several thousand examples like this.
2. benchmark some open-weight LMs on solving this task zero-shot (without reasoning tokens)
3. for a single model, create a causal mediation analysis experiment (patching from one run to another) to answer: "is there a hidden state layer that contains a representation of the running count of matching words, while processing the list of words?"
```

For (1) we can use synthetic data generation. You can find the types in `types.json` and the objects in `objects.json` (types are things like the type above: fruit, reptile, etc...) whereas objects are concrete objects (like dog, apple, cherry... above). Note that we are going to narrow this down a little bit, since English is in some sense hierarchical here (i.e. a Dog is a type of mammal which is a type of animal which is a type of living being, so if the type was "living being" then "mammal" would in English be a valid object: the lesson is clear, whether something is an object, type, or third class of words depends on what the type it would have to be under is). To avoid these issues, all objects are concrete (i.e. physical objects). `Edit: we are using one types2objects.json file to force GPT to actually generate an object for each type.`

For (2) we do greedy sampling. We also compare with OpenAI, Anthropic, and Gemini using the API (via LiteLLM). We save these as plots. We include BOTH thinking and non-thinking models (and pre-trained non-instruction-tuned models) where applicable. We plot a scaling plot as a function of number of parameters (we can basically consider the API models to be "infinite parameters" and a human to be that plus a tad).

For (3) I'm not entirely sure what to do. I suspect this may be in a 1D or 2D subspace. Below are some ideas (but let's assume that for each we have the following properties: (a) the correct answer will be outputted as a SINGLE TOKEN, (b) we have some set of hidden states we will consider ONLY, i.e. MLP or Residual stream, etc...):
1. Simply do attribution patching followed by activation patching to look for 1D, 2D (etc...) subspaces that when patched will flip the answer. Attribution could also just not be done. If we do attribution, we probably want subspaces that have high attribution to the correct answer token. We can use the patching to detect whether the flip is correct. According to Gemini I should consider integrated gradients: https://arxiv.org/pdf/1703.01365.
2. We generate the lists' prefixes in increasing length (basically the suffix of the prompt can be thought of as a logit-lens type thing, but instead of logit-lens it is prompt-lens). We ensure that the count increases correctly. Then I'm not sure what to do TBH, maybe use (3).
3. We do (2) but instead we have the same length but we flip one word or something. Then we take the difference of the activations and do a PCA, hoping to find very low rank. We then use the highest ranks and try them one by one doing perturbations.
4. We optimize low-rank adaptors (i.e. 1D or 2D) to flip the answer using gradient descent and then look for projections of the residual stream or hidden layer Basically we could use some linear combination of gradient ascent on the right number and gradient descent on the wrong answer.
5. I could use a linear probe to predict the final count and then see if any of those linear probes modulate the answer.
6. Calculate a "center of means" difference vector (or set, i.e. per activations) and then see which of the detected methods best flip the answer later on and make some sanity checks.

I think (1) is better than (4) because it's a linear version of the same thing but runs faster in theory. I think (2) and (3) could be interesting validation or visualization tools (i.e. check that the subspace grows over the course of the words) but I think they are not necessarily better than (1) for detecting the subspace. I think (5) and (6) are easier than (1).

Bonus things:
- Check if it is indeed an interpretable subspace (1D length or 2D circle, etc...)
- Check whether it is robust to different kinds of prompt formats
- Check if you change the thing you ask for and you do the ablation, does only the number change or also some contents?
- Consider more hookpoints. Specifically I am interested in **(1) something that is post-layer-norm, (2) attention values, attention Z, MLP up-projection**, though we could also do more.
- See if this applies to other forms of counting
- Do some literature research and compare to existing circuits work (I think Anthropic recently came out with something, for example)
- Augment our analysis to build up a _circuit_ by doing something akin to, possibly, ACDC (Gemini suggests looking into Meng's Causal Tracing)
- See if an SAE finds the direction (for this it would be smart to use a Gemmascope SAE)
- Try thinking models (i.e. deep seek: https://huggingface.co/deepseek-ai/DeepSeek-R1)
- Try instruction-tuned models
- Try more models: `mistralai/Mistral-7B-v0.3`, `mistralai/Mistral-7B-v0.1`, llama3 models (look at https://huggingface.co/meta-llama), microsoft phi models (look at https://huggingface.co/microsoft?search_models=phi), and qwen models (look at https://huggingface.co/Qwen)
- Try to do things on more tokens.

I will probably need to learn nnsight for this. Well... I don't need to but it might be nice TBH.

# Experiments (Running Log)
## Experiment 1: Linear Probes
**Data.** We will do part 1 using GPT-generated `type2object_gpt4o.json` file. We generate random combinations, specifying a-priori the number of objects that are correct amd the total number of objects (given a uniformly-at-random chosen type, we pick uniformly at random from the correct objects and then some other objects uniformly at random).

**Models.** For parts 2 and 3 we will check out the following models:
1. OpenAI 4o Mini (skyline; I only use Mini with good prompting since it's a lot cheapter than 4o)
2. GPT-2 small and xl: `openai-community/gpt2` and `openai-community/gpt2-xl`)
3. Gemmascope Gemma-2 models: `google/gemma-2-2b` and `google/gemma-2-9b` (just use the pre-trained model and do greedy generation with smart prompting). We choose to use these because pretrained is comparable to gpt2 and gemma-2 has gemma-scope (so later on if we want to include the )

As you can see, we are only focusing on base-models with good prompting for now. We will also ONLY be focusing on the residual stream and ONLY on the LAST TOKEN (i.e. the prediction token) for now.

For each model and prompt/parsing strategy (the tuple of model + prompt/parsing defines our experimental object) we create a "results" file that contains the following
1. Raw accuracy
2. For each layer, the accuracy of our best linear probe (i.e. dictionary of hookpoint to probe accuracy AND sorted list of hookpoints).

This is stored in a pydantic schema that has built-in functionality for creating the plots, etc... we want and storing the data we care about. Note that for gpt-4o we simply have an empty list for hookpoint sand an empty list for our linear probes. This gives us:
1. Per-model a heatmap/bar plot of best-performing hookpoints
2. A "scaling plot" of parameters in the x-axis and linear probe violin plot on the y axis (this is a regular violin plot but the models are sorted by size).

With this, we will have done parts 2 and 3 minimally. Then we will test the quality of our probes. These are the tests:
1. We filter for the ones where the accuracy is above a threshold (i.e. error is below 1 on average).
2. We will find a mapping from the linear regression prediction to the magnitude we estimated. We will flip the magnitude to random other values (within the range we tried on) and check if the prediction flips. We will filter for the ones where flipping successsfully flips over some amount (i.e. say 80%).

If this doesn't work we will make a new experiment and see where we get to. If this does work, we will look for more verification strategies.

FYI to do this, for each model we:
1. Collect and store the activations somewhere (using a cached dataset object that can choose to store or not)
2. Run linear regression
3. Store the results for accuracy; we can run the plotting stuff from a jupyter notebook. We also store the actual subspace/functional. So we can later do interventions.
4. Load the results in jupyter and try interventions.