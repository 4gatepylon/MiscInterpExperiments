Simple experiment to test for whether it's possible to predict and then extract the relevant circuit components to length.

Main things here are the following (for now):
1. Predict length from text (with numerical regression)
2. Predict length from residual stream (try averaging, try a couple tokens at random intervals based on distance from the end of the sequence).
3. Steer length on the reisudal stream while maintaining coherence

It may be the case that there is a lot of other random shit that is not actually used because I sort of set my scope to be too big (I planned to also predict, steer, and extract the lengths of certain tasks and other structural, planning, and "stateful" attributes). You should mainly look at the `scratchpad.ipynb` notebooks and HOPEFULLY I will get time to make a more interpretable one for you.

Make sure to install black for jupyter `black[jupyter]` and seaborn.

`WARNING: the dataset setup/tokenization is broken right now.`

XXX todos
0. Fix the bug and update the report
    - Should be able to enable batched generation: first make it fast: next step is to batch across datasets
    - Then make it correct (make sure to print out the questions, etc...)
    - Then be able to modify which tokens you apply this shit at: consider using nnsight as well (this
        is actually not a bad idea even if it's non-trial since I know i will almost certainly use this in the future)
1. Point to the drive from my website and hide most of the unwanted blogposts
2. Point to the drive from the github
3. Commit the weights or store them on drive and link them (not the weights but more importantly the steering vectors)
4. Make the website also include our DL project