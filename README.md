# Crossword Space: Latent Manifold Learning for Italian Crosswords and Beyond

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](img/archs.svg)


This is the public repository of our paper entitled: *Crossword Space: Latent Manifold Learning for Italian Crosswords and Beyond*, C.Ciaccio, G. Sarti, A. Miaschi, F. Dell'Orletta (CLiC-it 2025).

The repository contains the resources and code that we developed in order to run our experiments on crossword clues answering, investigating siamese and asymmetric dual encoder architectures and extending our analysis to neologisms in order to assess to what degree our systems can handle lexical innovations. Specifically we release the code for computing baselines, for instantiating and training both the dual encoder architectures.


If you use any of the following contents for your work, we kindly ask you to cite our paper:

```
```

> **Abstract:** Answering crossword puzzle clues presents a challenging retrieval task that requires matching linguistically rich and often
ambiguous clues with appropriate solutions. While traditional retrieval-based strategies can commonly be used to address this
issue, wordplays and other lateral thinking strategies limit the effectiveness of conventional lexical and semantic approaches.
In this work, we address the clue answering task as an information retrieval problem exploiting the potential of encoder-based
Transformer models to learn a shared latent space between clues and solutions. In particular, we propose for the first time
a collection of siamese and asymmetric dual encoder architectures trained to capture the complex properties and relation
characterizing crossword clues and their solutions for the Italian language. After comparing various architectures for this
task, we show that the strong retrieval capabilities of these systems extend to neologisms and dictionary terms, suggesting
their potential use in linguistic analyses beyond the scope of language games.