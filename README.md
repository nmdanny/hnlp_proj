# What this project is about

This project deals with stylometry of Hebrew texts for the purpose of authorship attribution.
Stylometry is a branch of linguistics dedicated to analyzing the style of texts, under the
assumption that each author writes in a consistent yet unique way. 

In some way, stylometric analysis is the dual of topic analysis: In topic analysis, we focus on the
content of a text, and try to find meaningful words that form a topic. In stylometry, we tend to do
the opposite, and focus on the function words(which are devoid of content), in hope of learning about the
writing style of the author.

Stylometry has various applications, for example:

- Analysis of historical documents and literary works(e.g, authorship of the Bible) 

- Detection of plagiarism and academic fraud 

- Legal purposes, for example, the infamous Unabomber was caught using stylometric analysis

- Detection of sock puppets(Internet users with multiple accounts) and troll farms

In this project, I will evaluate several algorithms for different tasks involving stylometry

# Methods

## [John Burrows' Delta Method](https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python#third-stylometric-test-john-burrows-delta-method-advanced)

This is one of the most popular algorithms for authorship attribution. It is used when the number of
authors is fixed, and we already have prior knowledge of texts attributed to these authors. Such
scenario can be seen as a classification task.

This algorithm was proposed by John F. Burrows' in 2002, and was later analyzed in a more ML centric way,
as a nearest neighbor classifier by [Shlomo Argamon in 2008](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.842.4317&rep=rep1&type=pdf) , I will explain it shortly in from the ML perspective:

As a supervised ML classification algorithm, there are two stages:

1. Fitting this classifier consists of merely defining and saving a proper feature matrix(this is a
   "lazy" learner), which we'll define as follows:
   
   - Fix `n` (number of features, hyper-parameter) and `x` number of authors 
   
   
   - Pick the `n` most frequent words(of all texts and all authors) as features 
   
   - Create a `x * n` feature matrix, where each feature is the share of the word within the
   author's subcorpora, in other words, the number of times the word appears within texts belonging
   to that author, divided by the total number of words in those texts.  
   
   - Standarize the feature matrix, so now the features are z-scores. This is done by subtracting
   each share from the population mean of the shares(over all authors) divided by the population
   standard deviation. 

   The intuitive meaning of these z-scores is how far is the author's usage of a certain feature(word) from the norm?

   In essence, we have done a "mean of means" (the first mean is when we divided the term
   frequency of an author's subcorpus, by the total term count in that subcorpus, the second is part
   of the standarization, done over all authors), thus ensuring that authors with a longer subcorpus
   do not influence the z-scores too much.


2. In order to evaluate an anonymous text, we apply the same transformation as above to embed the text
   within the space `n`-dimensional space, and then we calculate the distance between the anonymous
   text's vector, to the vectors of each of the authors, and pick the author with the closest vector.
   
   In the original paper, John burrows proposed his own metric ("Burrow's Delta"/Manhattan distance), but in principle any metric can be used (Euclidian distance, angular distance)

Conceptually, this is 1-nn using a 'brute' approach. Since each author/class has only 1 sample in the feature space
(the embedded vector represents all of that author's texts), there is no point in having a majority vote.
When dealing with very large amounts of authors, one can use more efficient nearest neighbor algorithm, such as a KD tree.


This model is also fairly interpretable - the vectors in the feature matrix essentially encode the literary signature
of each author/text, by indicating how far from the norm is the usage of various function words.

### Ideas for improvement

- Instead of operating on words, operate on lemmas/morphemes, this is especially important for Hebrew which
  doesn't have as many function words as English. 

- Evaluate different distance metrics 

- Play around with the `n` parameter

- Use different weighting schemes for the term frequency(e.g, log scaling, boolean TF)

- Normalize feature vectors?


## Hierarchial clustering

Hierarchial clustering can be used when the number of authors is unknown(as opposed to regular
clustering) and we don't know anything about the authorship of the texts. 