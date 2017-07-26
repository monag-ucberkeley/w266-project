# N-grams and Word Embeddings

## Learning Language Models of Literary Characters

Based on this paper http://cs224d.stanford.edu/reports/oguz.pdf:

1. We first learn the general language model of literary characters. This step will not consider the differences between different characters, rather learns a general model.
2. For each character define a set of projection parameters and train on individual characters rather than whole dataset.
3. Evaluate the language model of the individual characters.

For the first step, we will use the n-gram modeling & glove vectors.
