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
### 2.1 Data mining and visualisation
### 2.2 Linear regression
### 2.3 Decision tree
### 2.4 Support Vector Machine (SVM)
### 2.5 Recurrent neural network
## 3. Results and Discussion
## 4. Conclusion
## References
Benham, Adam M. “Protein Secretion and the EndoplasmicReticulum.” Cold Spring Harb Perspect Biol 4 (2012). Ferro-Novick, Susan and Brose, Nils. “Traffic control system within cells.” Nature 504 (2013): 98.  
Klatt, Stephan, and Zoltán Konthur. “Secretory signal peptide modification for optimized antibody-fragment expression-secretion in Leishmania tarentolae.” Microbial cell factories 11 (2012): 97.  
Miller, Jefferey H. “Protein Synthesis.” In Encyclopedia of Genetics, by Sydney and Millern Jefferey H. Brenner, 1567. New York: Academic Press, 2001.  
Peng Chong, Shi Chaoshuo, Cao Xue, Li Yu, Liu Fufeng, Lu Fuping. “Factors Influencing Recombinant Protein Secretion Efficiency in Gram-Positive Bacteria: Signal Peptide and Beyond.” Frontiers in Bioengineering and Biotechnology 7 (2019): 139.  
Stephanie J. Popa, Julien Villeneuve, Sarah Stewart, Esther Perez Garcia, Anna Petrunkina Harrison, Kevin Moreau. “ Genome-wide CRISPR screening identifies new regulators of glycoprotein secretion.” bioRxiv , 2019: 522334.  
Yoshinori Tsuchiya, Kazuki Morioka, Junsuke Shirai, Yuichi Yokomizo and Kazuo Yoshida. “Gene design of signal sequence for effective secretion of protein.” Nucleic Acids Research Supplenzent 3 (2003): 261 -262.  
Yoshitaka Shirasaki, Mai Yamagishi, Nobutake Suzuki, Kazushi Izawa, Asahi Nakahara, Jun Mizuno, Shuichi Shoji, Toshio Heike, Yoshie Harada, Ryuta Nishikomori, Osamu Ohara. “Real-time single-cell imaging of proteinsecretion.” Scientific Report 4 (2014): 4736.  
Orfanoudaki, G., Markaki, M., Chatzi, K. et al. MatureP: prediction of secreted proteins with exclusive information from their mature regions. Sci Rep 7, 3263 (2017). https://doi.org/10.1038/s41598-017-03557-4  
Almagro Armenteros, J.J., Tsirigos, K.D., Sønderby, C.K. et al. SignalP 5.0 improves signal peptide predictions using deep neural networks. Nat Biotechnol 37, 420–423 (2019). https://doi.org/10.1038/s41587-019-0036-z  
Loos, Maria, et al. "Structural basis of the sub-cellular topology landscape of Escherichia coli." Frontiers in microbiology 10 (2019): 1670.  
Morgat A, Lombardot T, Coudert E, Axelsen K, Neto TB, Gehant S, Bansal P, Bolleman J, Gasteiger E, de Castro E, Baratin D, Pozzato M, Xenarios I, Poux S, Redaschi N, Bridge A, UniProt Consortium. Enzyme annotation in UniProtKB using Rhea Bioinformatics (2019)  
Alley, Ethan C., et al. "Unified rational protein engineering with sequence-only deep representation learning." bioRxiv (2019): 589333.  
