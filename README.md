# Gene-Gene Interaction Detection with Deep Learning
Code for "Gene-Gene Interaction Detection with Deep Learning"

---
#### Toy example on simulated data:
1. Run "run_interaction.py" to estimate pairwise interaction scores between genes on the original dataset. Results are saved in "/InteractionScore".
2. Run "run_permutation.py" to estimate interaction scores between genes on 'num_run' (defined in the file) permutation dataset. Results are saved in "/PermutationDistribution". 
3. Repeat Step 2 multiple times (in parallel) to obtain a permutation distribution.

---
#### Use your own data:
- Replace "/data/genotype.csv" with your own genotype data with size (N, P), where each row represents each individual, and each column represents each SNP.
- Replace "/data/phenotype.csv" with your own phenotype data with size (N, 1).
- Replace "/data/snp_size.csv" with your own SNPs sizes data with size (M, 1), where each number represents the number of SNPs of each gene. The summation of all SNP sizes is P.
