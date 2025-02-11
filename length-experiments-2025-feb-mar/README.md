Simple experiment to test for whether it's possible to predict and then extract the relevant circuit components to length.

Main things here are the following (for now):
1. Predict length from text (with numerical regression)
2. Predict length from residual stream (try averaging, try a couple tokens at random intervals based on distance from the end of the sequence).
3. Steer length on the reisudal stream while maintaining coherence

It may be the case that there is a lot of other random shit that is not actually used because I sort of set my scope to be too big (I planned to also predict, steer, and extract the lengths of certain tasks and other structural, planning, and "stateful" attributes). You should mainly look at the `scratchpad.ipynb` notebooks and HOPEFULLY I will get time to make a more interpretable one for you.

Make sure to install black for jupyter `black[jupyter]` and seaborn.