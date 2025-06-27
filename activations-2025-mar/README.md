# A random idea
## Research Log
The idea to try to learn neural network architectures that  can learn different sort of taylor-like functions:
```
has anyone tried the following: imagine I want to be able to model functions that are sums of products powers of different variables: x, y, z, ... so like I want to be able to learn weights to learn ax^by^c.... + dx^ey^f.... with weights a, b, c, d, e, f, ....so I could do this:
vector of x, y, z, .... and then I take the logarithm to get ln(x) ln(y) etc... and then I matmul that by the power weight matrix W_p (no bias for now) and then I take the exponential of that and then I dot product it with a vector W_w... for example with pointwise ln and exp:W_w @ exp(W_p @ ln(x))) => gives me terms OF a taylor series where the NUMBER OF TERMS is bounded but the powers are not! has anyone done this?
```

According to Grok (because I happen to have lost deep research access elswhere ugh) product neural netwokrs might do this, but from looking at the paper it linked, that doesn't seem to be the exact same (they do some kind of outer product and the discussion seems to involve some trees)? According to perplexity, https://en.wikipedia.org/wiki/LogSumExp (basically softmax) activation functions exist and kind of... work (as in they are universal approximators).

Anthropic tried SoLU activation functions: https://transformer-circuits.pub/2022/solu/index.html#section-4 and give an argument that having superlinear functions would lead to less polysemanticity, because if the information was spread around multiple neurons (with the same magnitude) you would lose a lot of magnitude (i.e. spreading will shrink it). SoLU is literally pointwise `x * softmax(x)`. They also try with just `x * exp(x)` AND layer norm (which is the same as if you had applied layer norm on the softmax since it's variance normalized). It appears to lead to worse performance.

This is best fitted for physical laws, so, we should pick such a dataset to test probably.

Numerical stability seems to be a challenge.

More ideas:
- https://x.com/i/grok/share/sIpGGDBV4t1ba4uso7CfwWb80
- https://www.perplexity.ai/search/can-you-please-look-for-resear-4nm7BzV5QpioH8U7511LaA
- (only focuses on a taxonomy and interpretability of different activation functions) https://elicit.com/review/516f6e69-5eb3-46fd-ace0-283785d21b09

The main ideas/question (areas):
- New types of activations (pointwise)
- Multidimensional activations
- Dynamic convolutional kernels (i.e. first identify objects and then search for them around the image... I think it's KIND of like attention of a certain kind)
- Dyanmic weights in general (just curious about things other than ^)
- Posynomial/Polynomial (https://en.wikipedia.org/wiki/Posynomial) and different interaction forms for neural networks
- Iterative training

Also curious about whether we can enable different kinds of symmetries, but I did not look into this today.

Takeaways
- XXX ask about iterative training plz and scheduling gradient updates (basically we can call this class of techniques: "cooking")
- Posynomial and product networks seem hard to optimize (and people try genetic programming or particle swarm optimization or other optimization methods)
- It seems like you might be able to convert posynomial learning into a convex learning objective: https://optimization.cbe.cornell.edu/index.php?title=Geometric_programming
- Dynamic convolution neural networks have been tried: https://proceedings.neurips.cc/paper_files/paper/2016/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf
    - Code: https://github.com/dbbert/dfn (roughly 5 years old)
    - Question: would just inserting attention in the end be equivalent?
    - TODO(Adriano) read more about this, it seems pretty interesting... also as a general note I think the main benefit would be reduction in weights and improvements in generalization (i.e. some form of compression)

## Experiments and Results
TODO(Adriano)