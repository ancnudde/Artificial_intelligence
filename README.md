# Predicting cellular location using UniRep embedding of protein sequence

Buisson-Chavot Guillaume 
Cnudde Anthony
Ody Jessica
Van den Schilden Jan

30-06-2020

## 1. Introduction

<!--Part why knowing is important -->
Protein synthesis (Miller 2001) and subsequent subcellular localization involves complex mechanisms frequently studied over the years .
Many of these synthesized proteins are excreted into the extracellular environment (Benham 2012).
Such extracellular proteins allow cells to interact with the outside world or other cells.
Without, multicellular organisms would be unable intercellular interactions to form tissues and body parts (Yoshitaka Shirasaki 2014).
But also in bacterial cells there can be up to a third non-cytoplasmic proteins (Orfanoudaki 2017).
There is a lot of interest on how to distinguish secreted from cytoplasmic proteins,
especially for those expressing recombinant proteins.
This is often seen in the medical sector or for the industrial production protein (Peng Chong 2019).

<!--Part about Signal peptides-->
Signal peptides are short peptide sequences between 15 and 30 residues at the N-terminal end of the protein that are recognized by cell pathways to excrete the protein (Benham 2012, Klatt 2012).
During transit, the signal peptide is cleaved off the protein (Peng Chong 2019).
Even though signal peptides display low evolutionary sequential similarity,
the underlying biophysical properties display an high degree of conservation (Orfanoudaki 2017).
It consists of three regions: a positively charged region, a hydrophobic region, and a region with the cleavage site (Klatt 2012, Peng Chong 2019).
Software like SignalP-5.0 use deep neural networks to recognize these signal peptides and predict cellular localization (Armenteros 2019).

<!-- Part about other intrinsic factors-->
Even though signal peptides are often necessary for protein excretion,
they are in themselves insufficient.
Other features, 
such as secondary structure propensity,
dynamics,
and amino acid composition,
play an important role in determining protein location.
Predictors that use these features have been reported with an success rate of 95.5% success (Loos 2019).

<!-- Limitations of features -->
One limitation of including features is that the accuracy of the methods depends on how much information each feature can provide.
There might also be very important hidden features we don't know about,
but necessary for optimal performance.
On the other hand,
some features could be very descriptive but experimentally expensive to obtain.
Protein sequence information has become very cheap in the last years however.
As on the moment of writing, 
the public database UniProt (Morgat 2019) contained  181,252,700 sequences.
However, only 562,253 (0.32 %) contained manual annotation of features with experimental evidence.
Frequently when producing ML-based predictors,
these are based on well defined features and as a result discard more than 99% of the available sequences.
However just the knowledge that the sequence is from and extant protein is already useful information.

<!-- UniRep representation --> 
A research group (Alley 2019) took inspiration from state-of-art natural language processing method.
This method trains a mLSTM RNN a next character prediction problem,
and uses the hidden states to form a rich representation of the text.
Similarly, Alley et al. trained a mLSTM RNN on next residue prediction of the protein sequences.
To achieve this goal, 
the neural network will encode its own features in the hidden nodes to achieve his goal.
A trained network can subsequently be used to generate a fixed length vector in which important protein features are encoded.
They have shown that such vector can significantly improve the performance of ML based predictors as opposed to using raw sequence information.

<!--What we did-->
In this work,
The UniProt REST API was used to generate a dataset of cytoplasmic and periplasmic proteins.
After reducing the sampling bias with CD-HIT,
UniRep vectors were generated with the mLSTM RNN of Alley 2019 et al.
Using these fixed length vectors,
we trained our own implementation of
linear regression,
decision tree,
support vector machine (SVM),
and recurrent neural network.
The performance of the different methods was compared.

## 2. Methods

### 2.1. Data mining

The protein dataset for this work was generated using the [Uniprot REST API](https://www.uniprot.org/help/api%5Fqueries). 
The API provides a programmatic access to download the sequences through queries.
The goal in this work was to make a classifier that can distinguish between cytoplasmic and periplasmic proteins.
Only Gram-negative Bacteria have a periplasm,
but this is not one phylogenetic group.
In this work we limited ourselves to Gammaproteobacteria.

To generate a set of cytoplasmic proteins,
the query asked for proteins with an annotation of being located in the cytoplasm or cytosol.
As an extra safeguard,
it was specified that those proteins could not have an annotation of containing a signal peptide as cytoplasmic proteins normally do not have signal peptides.
The query is shown below.

```
QUERY="taxonomy:Gammaproteobacteria 
(locations:(location:cytoplasm) 
OR locations:(location:cytosol)) 
NOT annotation:(type:signal)"
```


To generate the set of periplasmic proteins,
a similar search was performed.
This time looking for an annotation of the protein being in the periplasm and the presence of a signal peptide (shown below).

```
QUERY="taxonomy:Gammaproteobacteria 
locations:(location:periplasm) 
annotation:(type:signal)"
```

As we explain before, signal peptides are used to identify destination of proteins[CITATION Ben12 \l 2060  \m Kla12 \m Pen19], 
but it has been noted that certain biophysical features are necessary in addition to guarantee its translocation.
Therefore, a third dataset was generated which was identical to the periplasm dataset, 
except that the signal peptides were cut off to compare the performance of the different ML methods when trained with and without signal peptide.

To limit sampling bias, 
the software CD-HIT a calculates the percentage identity between the protein sequences. 
Sequences with more identity than 50 percent are clustered together and a representative was chosen.

Finally, the sequences were transformed into a fixed length vector representation using UniRep. 
This method extracts states from an unsupervised trained mLSTM-RNN and combines them into a fixed length UniRep representation. 
This representation contains essential structural and functional features that can be used by ML algorithms to distinguish between Periplasmic and Cytoplasmic proteins.

For the decision tree and the neural network, and linear regression, 
3000 sequences of cytoplasmic protein and 3000 sequences of periplasmic protein are selected from the datasets previously shuffled to randomize the samples 
This gives a working dataset of 6000 sequences.