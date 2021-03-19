# Gene-Gene Interaction Detection with Deep Learning
Code for "Gene-Gene Interaction Detection with Deep Learning". We design a NN based approach to recover the interactions between genes, which we defined as sets of SNPs.

---
#### Toy example on simulated data:
On the simulated dataset, we simulate 20 SNPs ("/data/genotype.csv") from 2 genes (with sizes 12 and 8 respectivaly, shown in "/data/snp_size.csv"). We simulate the phenotype which has interaction between two genes.
1. Run "run_interaction.py" to estimate pairwise interaction scores between genes on the original dataset. Results are saved in "/InteractionScore/NN_i.csv".
2. Run "run_permutation.py" to estimate interaction scores between genes on 'num_run' (defined in the file) permutation dataset. Results are saved in "/PermutationDistribution/NN_i.csv". 

We suggest to run step 2 in parallel.

---
#### Use your own data:
- Replace "/data/genotype.csv" with your own genotype data with size (N, P), where each row represents each individual, and each column represents each SNP.
- Replace "/data/phenotype.csv" with your own phenotype data with size (N, 1).
- Replace "/data/snp_size.csv" with your own SNPs sizes data with size (M, 1), where each number represents the number of SNPs of each gene. The summation of all SNP sizes is P.
