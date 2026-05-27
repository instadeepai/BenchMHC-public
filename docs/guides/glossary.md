# Glossary

Short definitions for terms used across the documentation.

!!! tip "Target audience"

    It is aimed at readers who know:

    - immunology but not our ML notation,
    - or ML readers who need the biology context.

## Data layout and training modes

### Single-allelic (SA)

Data where each peptide–allele pair is observed in a context with **one** MHC allele resolved (e.g.
a single-allele cell line or comparable setup). Training and evaluation often use SA tables where
each row is one peptide with one allele.

### Multi-allelic (MA)

Data from samples that express **several** MHC alleles at once (e.g. multi-allele cell lines).
Peptides may be observed without knowing which allele presented them; models can use **bags** of
candidates per sample and iterative annotation (see the [Training](training.md) guide).

!!! example "SA vs MA"

    Here is how the data looks like for SA and MA with MHC1 samples:

    ```bash
    peptide,allele,hit
    AAAAAA,HLA__C1402;HLA__C1401;HLA__C1504,1.0  # Multi-allelic data
    BBBBBB,HLA__B4402,1.0                        # Single-allelic data
    ```

## Experimental readouts: BA, EL, HLA vs non-HLA

### Binding affinity (BA)

An **in vitro** measure of how tightly a peptide binds to an MHC molecule (often summarized as
IC50). BA data answers "does it bind?" rather than "was it naturally presented?".

### Eluted ligand (EL)

**Mass spectrometry (MS)**–identified peptides that were **actually presented** on MHC at the cell
surface (or eluted from MHC). EL data reflects biological presentation more directly than BA alone.

### HLA vs non-HLA

**HLA** (human leukocyte antigen) refers to human MHC molecules. **Non-HLA** denotes other species'
MHC alleles in the same pipelines. Dataset docs often report counts for **HLA-only** subsets versus
all alleles.

## Molecules and complexes

### pMHC

**Peptide–MHC complex**: a peptide bound to an MHC molecule. Presentation prediction targets whether
a given peptide forms a relevant pMHC on the cell surface.

### MHC class I vs II

**MHC-I** typically presents shorter peptides (often centered on a **9mer core** for class I).
**MHC-II** presents longer, more variable-length peptides. BenchMHC materials may refer to **MHC1**
/ **MHC2** in model names and configs.

## Features used by predictors

### Pseudo-sequence

A fixed-length amino acid sequence that **represents** an MHC allele for modeling (e.g. positions
inferred from pocket residues), so a neural model can consume "allele" as a sequence.

#### NetMHCpan pseudo-sequence

From ["NetMHCpan, a method for MHC class I binding prediction beyond humans", Hoof, 2008](
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3319061/), the NetMHCpan pseudo-sequence is defined as
below and can be considered as taking the polymorphic residues of the MHC in contact with the
peptide:

> The MHC class I molecule was represented by a pseudo-sequence consisting of amino acid residues
> in contact with the peptide. The contact residues are defined as being within 4.0 Å of the peptide
> in any of a representative set of HLA-A and HLA-B structures binding a nonameric peptide. Of all
> contact residues, only those that were polymorphic in any known HLA-A, HLA-B, and HLA-C protein
> sequence were included, giving rise to a pseudo-sequence consisting of 34 amino acid residues
> (Nielsen et al. 2007). This pseudo-sequence mapping was applied to all MHC molecules in this study.
> This could lead us to discard essential peptide–MHC interactions for non-classical and non-human
> MHC molecules. However, no quantitative peptide-binding data are available for non-classical HLA
> molecules, and only very limited data are available for non-human primates. The pan-specific
> approach relies on the ability of the neural networks to capture general features of the
> relationship between peptides and HLA pseudo-sequences and interpret these in terms of binding
> affinity. Only interactions that are polymorphic in the training data can aid the neural network
> learning. It would hence not be possible for the NetMHCpan method to learn from such extended
> pseudo-sequence mappings due to the lack of polymorphism at the extended MHC positions in the
> training data.

### 9mer core

For many **MHC-I** predictors, the core **nine** amino acids that sit in the binding groove.

### NNAlign

A family of models / features (NetMHCpan-style) that use **neural networks** on peptide and allele
encodings. The `compute-nnalign-features` command adds NNAlign-oriented peptide features for
compatible models.

## Metrics (evaluation)

### FRANK score

Within each evaluation **group** (e.g. one true peptide plus its decoys), the fraction of **decoys**
whose predicted score is **higher** than the **hit** (true peptide). It ranges from 0 to 1;
**lower is better**. Used notably for **CD8 epitopes**–style benchmarks; see also model pages such
as [NetMHCpan-4.1](../models/netmhcpan41_v_0_0_0.md).

### Top-K

It computes how many of the true positives appear among the **K** highest-scoring predicted
probabilities in the group. We often chose K equal to the number of true positives in the group.
It ranges from 0 to 1; **higher is better**.

### Other metrics

`evaluate` reports metrics such as **PR-AUC**, **AP**, **ROC-AUC**, **Top-K**, etc.
See the [Inference](inference.md) guide and the `evaluate` CLI help.

## Labels in this repository

### Hit

A **positive** example: a peptide treated as genuinely presented (or binding, depending on the task)
for that allele or sample. In tables this is often a binary **hit** column or derived from EL/BA
definitions in the dataset.

### Decoy

A **negative** or competitor peptide: often sampled from proteomes to match the hit in
length/context but not labeled as presented. `generate-decoys` builds decoys from hits for training
or evaluation.

### Hit vs decoy in the pipeline

Workflows typically start from **hits**, optionally enrich with protein context
(`assign-protein-features`), then **`generate-decoys`** to add negatives. Models are trained and
evaluated on mixes of hits and decoys.
