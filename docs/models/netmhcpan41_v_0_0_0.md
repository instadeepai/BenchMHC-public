---
tags:
  - Denmark Technical University - DTU
  - MHC1
  - HLA / BA
  - BA
  - HLA / SA
  - SA
  - HLA / MA
  - MA
---

# NetMHCpan-4.1 - v0.0.0

NetMHCpan-4.1[^1] is a **peptide-MHC1 presentation predictor** developed by the Denmark Technical
University (DTU).

It is based on multiple iterations of research, see the following "version history" (a more detailed
version of it is available in the [`Version history` tab of the official page of
NetMHCpan.4-1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)):

1. 1.0[^2] (2007): BA (binding affinity) data for HLA-A/B
1. 1.1 (2007): + predictions with 8-11mers
1. 2.0 (2008): + HLA-C/E/G and non-HLAs
1. 2.1 (2009): + percentile ranks for per-allele calibration
1. 2.2/2.3/2.4/2.8 (2009): new data only
1. 3.0[^3] (2016): + training on 8-13-length peptides thanks to a new alignment method with indels
   (NNAlign)
1. 4.0 (2017): + training on 14-length peptides  + EL SA (eluted ligand single-allelic) data with
   two outputs: likelihood of ligand presentation and binding affinity
1. 4.1[^6] (2019): + EL MA (eluted ligand multi-allelic) data thanks to NNAlign_MA (**this model**)
1. NetMHCpanExp-1.0[^7] (2022): + gene expression data

!!! info "Similarities and differences between original and reproduced NetMHCpan-4.1"

    The following table indicates whether technical details of the reproduced model are:

    - :green_circle: similar to the original model,

    - :yellow_circle: maybe different from the original model, as information was
      lacking in the published literature,

    - :red_circle: different from the original model by choice.

    | Technical details                                                      | Status          |
    |------------------------------------------------------------------------|-----------------|
    | [NNAlign technique](#technical-nn-align)                               | :green_circle:  |
    | [Ensemble of 50 models](#technical-ensemble)                           | :green_circle:  |
    | [Hidden size of the MLP](#technical-hidden-size)                       | :yellow_circle: |
    | [MLP weight initialization](#technical-mlp-initialization)             | :yellow_circle: |
    | [Input features](#technical-features)                                  | :green_circle:  |
    | [BLOSUM62 amino acid encoding](#technical-blosum62)                    | :yellow_circle: |
    | [Outputs](#technical-outputs)                                          | :green_circle:  |
    | [Instance-based max pooling over 9mer cores](#technical-instance-mil)  | :green_circle:  |
    | [9mer core selection for BA](#technical-ba-score)                      | :yellow_circle: |
    | [Loss function](#technical-loss)                                       | :yellow_circle: |
    | [Optimizer](#technical-sgd)                                            | :green_circle:  |
    | [Number of epochs](#technical-epochs)                                  | :green_circle:  |
    | [Early stopping](#technical-early-stopping)                            | :yellow_circle: |
    | [Batch size](#technical-batch-size)                                    | :yellow_circle: |
    | [Batch balancing](#technical-batch-balancing)                          | :red_circle:    |
    | [Iterative annotation on MA data](#technical-iterative-annotation)     | :green_circle:  |
    | [Prediction score rescaling](#technical-prediction-score-rescaling)    | :green_circle:  |
    | [SA + MA validation data](#technical-ma-validation-data)               | :red_circle:    |
    | [Training data](#technical-training-data)                              | :green_circle:  |
    | [Evaluation data](#technical-evaluation-data)                          | :green_circle:  |

---

## Architecture

### NNAlign

!!! quote "NNalign[^4]"

    NNAlign, based on artificial neural networks (ANNs), was first developed in 2009 for the
    prediction of peptide–MHC class II binding affinity […]  NNAlign has become the engine of some
    of the most successful tools for the prediction of peptide binding to MHC molecules, including
    NetMHC, NetMHCpan, NetMHCII and NetMHCIIpan.

#### Preprocessing

- :green_circle: <a id="technical-nn-align"></a> We use the NNAlign technique which transform any
  k-mer into all possible 9mer "cores" by either inserting consecutive `X` tokens or by deleting
  consecutive amino acids to all possible solutions.

![Preprocessing](../media/models/netmhcpan41_v_0_0_0/nn_align_preprocessing.png#only-light)
![Preprocessing](../media/models/netmhcpan41_v_0_0_0/nn_align_preprocessing_dark.png#only-dark)
/// caption
Preprocessing in NNAlign
///

!!! quote "9mer cores[^3]"

    Peptides longer than nine amino acids were reduced to a core of nine amino acids by applying
    consecutive amino acid deletions. These included both deletions at the end terminals and
    consecutive deletions within the peptide. In the case of peptides shorter than nine amino acids,
    a wildcard amino acid X (encoded as a vector of zeros) was inserted to extend the peptide to a
    9mer core. Deletions and insertions were attempted at all possible locations within the peptide
    and the configuration returning the highest predicted score was saved as the optimal binding
    core. The current best solution was used together with the MHC pseudo sequence for error
    back-propagation and the procedure was iterated.

#### Model

- :green_circle: <a id="technical-ensemble"></a> We train an ensemble of multi-layer perceptron
  (MLP) models of 1 hidden layer of either 56 or 66 as hidden size. 10 models are trained per
  training/validation split with a different random seed, resulting in a total of 50 models in the
  ensemble.

!!! warning ":yellow_circle: <a id="technical-hidden-size"></a> Hidden sizes"

    In NetMHCpan-4.1[^6], hidden sizes of 55 and 66 are mentioned _"The model architecture and
    training parameters were equal to those defined earlier (Reynisson et al., 2021). The complete
    model consisted of an ensemble of 50 networks with 56 and 66 hidden neurons with 5 random weight
    initializations for each of the 5 cross-validation folds (2 architectures, 5 seeds and
    5-folds)._" while in NetMHCpanExp[^7], 56 and 66 are mentioned: _"A total of 10 random seeds
    for weight initialization were used; the hidden layer was populated with 55 and 66 hidden
    neurons; and the training data was split into 5 partitions for cross-validation using a
    Hobohm1-based common motif algorithm (29) with a motif length of 8 amino acids. This yielded a
    final ensemble of 50 networks. All networks in the ensemble were trained using 200 iterations,
    with a burn-in period of 20 iterations and early stopping."_. We sticked to the latter (56
    and 66) but we do not expect a strong difference due to this choice.

- :yellow_circle: <a id="technical-mlp-initialization"></a> We initialize the weights of the MLP with
[**Kaiming Uniform initialization**](https://arxiv.org/abs/1502.01852).

- :green_circle: <a id="technical-features"></a> We use the features described in the
  "Implementation/Network training and architecture" section of NetMHCpan3.0[^3]:

  - Sequence of 43 AAs = 860 input neurons (as 20 amino acids):
    - 9-mer: sequence encoded in BLOSUM
      - BLOSUM62 encoding: :yellow_circle: <a id="technical-blosum62"></a> The exact matrix used is
        not clear in the literature, we used the BLOSUM62 one. It is derived from a large set of
        protein alignments and is widely used for protein sequence comparison and alignment.
      - Wildcard token `X` and unknown tokens are encoded as a 20-length 0 vector
    - Allele: pseudo-sequence encoded into a 34-mer sequence
  - Length of insertion = 1 input neuron w/ `[0, 1]`
  - Length of deletion = 1 input neuron w/ `[0, …, 6]`
  - Length of peptide flanking regions (which are larger than zero in the case of a predicted
    extension of the peptide outside either terminus of the binding groove) = 2 input neurons w/
    values in `[0, …, 5]`
  - Length L of the peptide encoded in 4 neurons: L≤8, L=9, L=10, L≥11
  - In total: **868 input neurons.**

- :green_circle: <a id="technical-outputs"></a> The output consists in 2 neurons, one for **BA**
  (binding affinity, `binding_affinity`) and one for **EL** (eluted ligand, `hit`).

![Architecture](../media/models/netmhcpan41_v_0_0_0/nn_align_architecture.png#only-light)
![Architecture](../media/models/netmhcpan41_v_0_0_0/nn_align_architecture_dark.png#only-dark)
/// caption
Architecture
///

!!! quote "Pseudo-sequence[^2]"

    The HLA sequence was encoded in terms of a pseudo-sequence
    consisting of amino acid residues in contact with the peptide. The
    contact residues are defined as being within 4.0 A˚ of the peptide in
    any of a representative set of HLA-A and -B structures with
    nonamer peptides. Only polymorphic residues from A, B, and C
    alleles were included giving rise to a pseudo-sequence consisting of
    34 amino acid residues

!!! tip "Pseudo-sequence mapping"

    A mapping from allele names to their pseudo-sequences is available in
    `data/mappings/allele2netmhc_pseudo_seq.json`.

!!! info "BLOSUM62"

    The BLOSUM62 matrix assigns scores to each pair of amino acids based on their substitution
    probabilities in aligned protein sequences. Positive scores indicate that the substitution is
    relatively common, while negative scores indicate that the substitution is relatively rare.

---

## Training

### Instance-based multiple instance learning over 9mer cores

- :green_circle: <a id="technical-instance-mil"></a> We use the original approach which consists in
  doing the back-propagation with the 9mer core having the highest output score out of the MLP. This
  corresponds to an **instance-based max pooling** using the multiple instance learning paradigm,
  with the instances being the 9mer cores of the peptide.
  - :yellow_circle: <a id="technical-ba-score"></a> It is not specified in the NetMHCpan literature
    whether the selected BA score for backpropagation is the maximal one over 9mer cores or the one
    associated to the 9mer core yielding the best EL score. We decided to stick to the former: the
    model is hence free to select different 9mer cores for BA and EL.

### Hyper-parameters

- :yellow_circle: <a id="technical-loss"></a> We use **binary cross-entropy (BCE)** over EL samples
  and **mean squared error (MSE)** over BA samples. The final loss is simply the addition of both
  losses.

!!! info "BA data encoding"

    Target value of the binding affinity of the training example rescaled between 0 and 1 using the
    relationship `1-log(aff)/log(50,000)` where `aff` is the IC50 affinity value in nM units

- :green_circle: <a id="technical-sgd"></a> We use **stochastic gradient descent (SGD)** with a
  fixed learning rate of **0.005**.
- :green_circle: <a id="technical-epochs"></a> We train models for **200 epochs**. :yellow_circle:
  <a id="technical-early-stopping"></a> The following early stopping has been used:
  - HLA / BA: early stopping on validation loss with patience of 20 epochs,
  - BA: early stopping on validation loss with patience of 20 epochs,
  - HLA / SA: early stopping on validation average precision (on EL) with patience of 20 epochs,
  - SA: early stopping on validation average precision (on EL) with patience of 20 epochs,
  - HLA / BA + SA: early stopping on validation average precision (on EL) with patience of
    :warning: 10 epochs,
  - BA + SA: early stopping on validation average precision (on EL) with patience of 20 epochs,
  - HLA / BA + SA + MA: early stopping on validation loss with patience of 20 epochs,
  - BA + SA + MA: early stopping on validation loss with patience of 20 epochs.
- :yellow_circle: <a id="technical-batch-size"></a> We use a **batch size of 1024**. [Different
  batch sizes (1024, 2048, 4096, 8192) were
  tested](https://github.com/instadeepai/BenchMHC/issues/46#issuecomment-2664993211). The
  per-step convergence rate was similar across the different batch sizes, hence we choose the
  smallest batch size (1024) to have more steps per epoch and hence speed up per-epoch convergence.
  This means that using lower batch sizes does not decrease the quality of the gradients and enable
  the same convergence rate per step as with higher batch sizes.
- :red_circle: <a id="technical-batch-balancing"></a> The original approach balances the batches so
  that the EL:BA ratio equals 1:1. **We do not balance the batch**, hence the per-batch EL:BA ratio
  is approximately equal to the original EL:BA ratio in the data. Early experiments suggested a
  negligible impact but we will directly assess this in an ablation study.

### Iterative annotation for (multi-allelic) MA data integration

- :green_circle: <a id="technical-iterative-annotation"></a> For models trained on MA data, we use
  the **iterative annotation** proposed in the original approach. First, the models are trained for
  20 epochs on SA data only (SA pre-training phase), then the models are trained on SA + MA data
  (SA + MA fine-tuning phase). At each epoch, the positive MA data is annotated with the latest
  checkpoint (i.e. one allele is selected for each (genotype, peptide) pair), the negative MA data
  is randomly annotated (i.e. one random allele is selected for each (genotype, peptide) pair), and
  the MA data is added to the SA data.

!!! quote "MA data integration[^5]"

    The MA extension of NNAlign consists of various critical steps. First, a neural network is
    pre-trained on SA data only during a burn-in period, using the NNAlign framework. This results
    in a pan-specific model with potential to infer binding specificities also for MHC molecules
    not included in the SA data set. After this initial training period (from now on referred to
    as “pre-training”), the data in the MA data sets are annotated. That is, binding for each
    positive peptide in the MA data set is predicted (using the ligand likelihood prediction value
    from the pre-trained model) to all the possible MHC molecules of the given cell line and the
    restriction is inferred from the highest prediction value. For negative MA data, a random MHC
    molecule from the given cell line is tagged. Next, the SA and now single-MHC annotated MA data
    are merged, and the model is retrained on the combined data. Note, that the MHC allele
    annotation is updated at each iteration and will in general change as the training progresses.

- :green_circle: <a id="technical-prediction-score-rescaling"></a> We use **prediction score
  rescaling** as detailed in the original approach, with the exact same hyper-parameters. Details on
  the reference set can be found [here](
  https://github.com/instadeepai/BenchMHC/issues/75#issuecomment-2733493959).

!!! quote "Prediction score rescaling[^5]"

    To level out differences in the prediction scores between MHC alleles imposed by the differences
    in number of positive training examples and distance to the training data included in the SA
    data set, a rescaling of the raw prediction values was implemented and applied in the MA data
    annotation. The rescaling was implemented as a z-score transformation of the raw prediction
    values using the relation `z  = (p - p_bar / sigma_bar)`, where `p` is the raw prediction value
    of the peptide to a given MHC molecules, and `p_bar` and `sigma_bar` are the mean and standard
    deviation of the distribution of prediction values for random natural peptides for the MHC
    molecule. Here, the score distribution was estimated by predicting binding of 10,000 random
    natural 9mer peptides to MHC molecule in question. Next, the mean and standard deviation were
    estimated from a positive normal distribution, iteratively excluding outliers
    (`z-score >=3` or `z-score <= -3`). [...] This estimation of `p_bar` and `sigma_bar` was
    repeated in each iteration round before annotating the MA data.  As the rescaling is imposed to
    level out score differences between MHC molecules characterized in the SA training binding data
    and molecules from the MA data distant to the training data, the need for rescaling should be
    leveled out as the MA data are included in the training and the NNAlign_MA training progresses.
    To achieve  this, the values of `p_bar` and sigma_bar were modified to converge towards uniform
    values `p_u` and `sigma_u` defined as the average of `p_bar` and `sigma_bar` over all molecules
    in the MA data set. This convergence was defined as `p_tilde = w * p_bar+ (1 - w) * p_u` and
    `sigma_tilde = w * sigma_bar + (1 - w) * sigma_u`, where `w = 1/(1 + exp((x-75)/10))` and `x`
    is the number of training iterations. With this relation, when `w` is close to 1 after
    pre-training (`x = 20`), the terms `p_u` and `sigma_u` vanish; on the other hand, as `x` passes
    100 iterations, `w` converges to 0 and the terms `p_bar` and `sigma_bar` will vanish. With this,
    one can modulate the rescaling of the data as a function of the iterations and the type of data
    being used for training (SA or MA). The shift value of the exponential present in `w` (75) is a
    tunable parameter that defines this adjustment schedule. Similar results as the ones shown in
    this work were obtained varying this value in the range [50, 100] (data not shown).

![Shift value](../media/models/netmhcpan41_v_0_0_0/w_shift_parameter.png#only-light)
![Shift value](../media/models/netmhcpan41_v_0_0_0/w_shift_parameter_dark.png#only-dark)
/// caption
Convergence scale parameter (75 is used in NNAlign_MA and in our reproduction).
///

![Mean and variance of p_tilde and sigma_tilder over
training](../media/models/netmhcpan41_v_0_0_0/p_tilde_sigma_tilde_over_split_0.png#only-light)
![Mean and
variance of p_tilde and sigma_tilder over
training](../media/models/netmhcpan41_v_0_0_0/p_tilde_sigma_tilde_over_split_0_dark.png#only-dark)
/// caption
Mean and variance of `p_tilde` and `sigma_tilde` across alleles over the course of the
training of one base BA + SA + MA model on split 0. We can observe that `p_tilde` and `sigma_tilde`
converge to uniform values thanks to the convergence scale parameter.
///

- :red_circle: <a id="technical-ma-validation-data"></a> **We only use SA data for validation
  data.**

---

## Training data

:green_circle: <a id="technical-training-data"></a> We used the **exact same [training
data](https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/)** as in the
original paper. For detailed information about the training data, see the
[MS Ligands - Training](../datasets/ms_ligands.md#training) section in the datasets documentation.

---

## Evaluation data

:green_circle: <a id="technical-evaluation-data"></a> We use the **[exact same evaluation
data](https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/)** as in the
original paper.

### MS ligands

The HLA/SA MS ligands evaluation set from NetMHCpan-4.1[^6] is used to benchmark our models. For
detailed information about the MS ligands evaluation data, see the
[MS Ligands - Evaluation](../datasets/ms_ligands.md#evaluation) section in the datasets
documentation.

The models’ performance on this dataset is reported using several metrics: positive predictive value
(PPV), average precision (AP), and area under the receiver operating characteristic curve (ROC-AUC).
Those metrics are computed globally on the entire set. Additionally, the PPV is also reported
per-allele. In general, PPV refers to the fraction of true positives among the assigned positives
and thus requires a rank threshold to define the latter. Here, we use a slightly different
definition of PPV (also named Top-`K` in the study) that depends on the total number of positives in
the data set. More precisely, to calculate the per-allele PPV, we retrieve the top-`K` ranked
peptides, where `K` is the number of true binding peptides for each allele, and measure how many of
these peptides are relevant. This is equivalent to recall@`K`. The mean PPV stands for the mean over
all the per-allele PPVs. In the case of global PPV, we retrieve the `K` binding peptides in the
entire data set. The distribution of the per-allele PPVs is also reported with bar- and box- (or Box
and Whisker) plots. The latter consists of a rectangular box with lines (called whiskers) extending
from either end. The box represents the middle 50% of the data, which includes the median (the
middle value of the data), the upper and lower quartiles (the 25th and 75th percentiles), and any
outliers (values that are significantly higher or lower than the majority of the data). The whiskers
extend to the minimum and maximum values of the data, unless there are outliers, in which case the
whiskers extend to the most extreme data value that is not an outlier.

### CD8 epitopes

The second evaluation set from NetMHCpan-4.1[^6], CD8 epitopes, is used.
For detailed information about the CD8 epitopes evaluation data, see the
[CD8 Epitopes - Evaluation](../datasets/cd8_epitopes.md#evaluation) section in the datasets
documentation.

The predictive performance on this dataset is evaluated with the
FRANK score. The latter measures how well a model can distinguish the true peptide (i.e. the
epitope) from generated decoy peptides for a given allele-peptide binding pair, with a lower score
indicating better performance. The score is calculated as the percentile rank of the true peptide's
prediction score among the set of decoys, and hence takes values between 0 and 1.

\[
\text{FRANK score} = \frac{\text{Number of decoys with a score higher than the hit}}{\text{Total number
 of decoys}}
\]

---

## Compute

Three different compute environments were used, depending on availability:

- Ubuntu on a VM w/ 104GB RAM 16 CPUs (w/ or w/o 1 16GB P100 GPU),
- Ubuntu on a VM w/ 208GB RAM 32 CPUs (w/ 1 16GB V100 GPU),
- MacOS on a local Mac w/ 20GB RAM 8 CPUs (w/ or w/o Apple Silicon M4).

In practice, probably because the training pipeline is mostly CPU-bound, it was quicker to train
without GPUs on the VM w/ 104GB RAM 16 CPUs (w/ or w/o 1 16GB P100 GPU). This was not observed on
the VM w/ 208GB RAM 32 CPUs (w/ 1 16GB V100 GPU). The third setup has only been used for _(HLA) BA_
and _BA_ models. Here again, using the CPU only was faster than using the integrated GPU.

---

## Models

!!! info "Data paths"

    Here are the model paths for each types of data:

    - (HLA) BA: `gs://bench-mhc/data/v0.0.0/training/ba_hla/`
    - BA: `gs://bench-mhc/data/v0.0.0/training/ba/`
    - (HLA) SA: `gs://bench-mhc/data/v0.0.0/training/sa_hla/`
    - SA: `gs://bench-mhc/data/v0.0.0/training/sa/`
    - (HLA) BA + SA: `gs://bench-mhc/data/v0.0.0/training/ba_sa_hla/`
    - BA + SA: `gs://bench-mhc/data/v0.0.0/training/ba_sa/`
    - (HLA) MA: `gs://bench-mhc/data/v0.0.0/training/ma_hla/`
    - MA: `gs://bench-mhc/data/v0.0.0/training/ma/`

    Based on their names, the following models have been trained using these datasets.

### (HLA) BA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_hla_ba_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - (HLA) BA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba.png#only-light)
![Training curves - (HLA) BA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba_dark.png#only-dark)
/// caption
Training curves - (HLA) BA
///

- Average time per single training = 666s = ~11mn

![Epochs over relative time - (HLA) BA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba.png#only-light)
![Epochs over relative time - (HLA) BA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba_dark.png#only-dark)
/// caption
Epochs over relative time - (HLA) BA
///

- This model has been trained in the following issue:
  [#64](https://github.com/instadeepai/BenchMHC/issues/64).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_hla_ba_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba.yml \
  -t data/v0.0.0/training/ba_hla/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba_hla/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### BA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_ba_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - BA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba.png#only-light)
![Training curves - BA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba_dark.png#only-dark)
/// caption
Training curves - BA
///

- Average time per single training = 824s = ~14mn

![Epochs over relative time - BA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba.png#only-light)
![Epochs over relative time - BA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba_dark.png#only-dark)
/// caption
Epochs over relative time - BA
///

- This model has been trained in the following issue:
  [#64](https://github.com/instadeepai/BenchMHC/issues/64).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_ba_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba.yml \
  -t data/v0.0.0/training/ba/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### (HLA) SA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_hla_sa_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - (HLA) SA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_sa.png#only-light)
![Training curves - (HLA) SA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_sa_dark.png#only-dark)
/// caption
Training curves - (HLA) SA
///

- Average time per single training = 32531s = ~9h2mn

![Epochs over relative time - (HLA) SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_sa.png#only-light)
![Epochs over relative time - (HLA) SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_sa_dark.png#only-dark)
/// caption
Epochs over relative time - (HLA) SA
///

- This model has been trained in the following issue:
  [#63](https://github.com/instadeepai/BenchMHC/issues/63).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_hla_sa_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_sa.yml \
  -t data/v0.0.0/training/sa_hla/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/sa_hla/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### SA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_sa_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - SA](../media/models/netmhcpan41_v_0_0_0/training_curves_sa.png#only-light)
![Training curves - SA](../media/models/netmhcpan41_v_0_0_0/training_curves_sa_dark.png#only-dark)
/// caption
Training curves - SA
///

- Average time per single training = 43222s = ~12h

![Epochs over relative time - SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_sa.png#only-light)
![Epochs over relative time - SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_sa_dark.png#only-dark)
/// caption
Epochs over relative time - SA
///

- This model has been trained in the following issue:
  [#63](https://github.com/instadeepai/BenchMHC/issues/63).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_sa_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_sa.yml \
  -t data/v0.0.0/training/sa/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/sa/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### (HLA) BA + SA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_hla_ba_sa_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - (HLA) BA + SA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba_sa.png#only-light)
![Training curves - (HLA) BA + SA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba_sa_dark.png#only-dark)
/// caption
Training curves - (HLA) BA + SA
///

- Average time per single training = 24391s = ~6h47mn

![Epochs over relative time - (HLA) BA + SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba_sa.png#only-light)
![Epochs over relative time - (HLA) BA + SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba_sa_dark.png#only-dark)
/// caption
Epochs over relative time - (HLA) BA + SA
///

- This model has been trained in the following issue:
  [#49](https://github.com/instadeepai/BenchMHC/issues/49).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_hla_ba_sa_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba_sa.yml \
  -t data/v0.0.0/training/ba_sa_hla/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba_sa_hla/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### BA + SA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_ba_sa_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - BA + SA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba_sa.png#only-light)
![Training curves - BA + SA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba_sa_dark.png#only-dark)
/// caption
Training curves - BA + SA
///

- Average time per single training = 51694s = ~14h22mn

![Epochs over relative time - BA + SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba_sa.png#only-light)
![Epochs over relative time - BA + SA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba_sa_dark.png#only-dark)
/// caption
Epochs over relative time - BA + SA
///

- This model has been trained in the following issue:
  [#53](https://github.com/instadeepai/BenchMHC/issues/53).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_ba_sa_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba_sa.yml \
  -t data/v0.0.0/training/ba_sa/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba_sa/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL>
  ```

### (HLA) BA + SA + MA

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_hla_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - (HLA) BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba_sa_ma.png#only-light)
![Training curves - (HLA) BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/training_curves_hla_ba_sa_ma_dark.png#only-dark)
/// caption
Training curves - (HLA) BA + SA + MA
///

- Average time per single training = 64189s = ~17h50mn

![Epochs over relative time - (HLA) BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba_sa_ma.png#only-light)
![Epochs over relative time - (HLA) BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_hla_ba_sa_ma_dark.png#only-dark)
/// caption
Epochs over relative time - (HLA) BA + SA + MA
///

- This model has been trained in the following issue:
  [#75](https://github.com/instadeepai/BenchMHC/issues/75).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_hla_ba_sa_ma_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba_sa.yml \
  -t data/v0.0.0/training/ba_sa_hla/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba_sa_hla/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -ma_t data/v0.0.0/training/ma_hla/split_<SPLIT>_train.csv \
  --use_prediction_score_rescaling \
  --reference_path data/reference/10k_9mers.csv \
  --sa_warmup_epochs 20 \
  --deconvolution_identifier MA_bag_identifier
  ```

### BA + SA + MA

!!! note "Final model"

    BA + SA + MA corresponds to our reproduction of NetMHCpan-4.1 as it has been trained on the
    entire training set from NetMHCpan-4.1.

- Ensemble model:

```bash
gs://bench-mhc/models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt
```

![Training curves - BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba_sa_ma.png#only-light)
![Training curves - BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/training_curves_ba_sa_ma_dark.png#only-dark)
/// caption
Training curves - BA + SA + MA
///

- Average time per single training = 97207s = ~27h7mn

![Epochs over relative time - BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba_sa_ma.png#only-light)
![Epochs over relative time - BA + SA + MA](../media/models/netmhcpan41_v_0_0_0/epochs_over_time_ba_sa_ma_dark.png#only-dark)
/// caption
Epochs over relative time - BA + SA + MA
///

- This model has been trained in the following issue:
  [#75](https://github.com/instadeepai/BenchMHC/issues/75).

- A submodel of this ensemble model has been trained the following command:

  ```bash
  train \
  -n nnalign_mhc1_ba_sa_ma_split_<SPLIT>_bs_1024_sgd_5en2_rs_<SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -c configuration/nnalign_ba_sa.yml -t data/v0.0.0/training/ba_sa/split_<SPLIT>_train.csv \
  -v data/v0.0.0/training/ba_sa/split_<SPLIT>_tune.csv \
  -rs <SPLIT><2_DIGIT_NUM_SUBMODEL> \
  -ma_t data/v0.0.0/training/ma/split_<SPLIT>_train.csv \
  --use_prediction_score_rescaling \
  --reference_path data/reference/10k_9mers.csv \
  --sa_warmup_epochs 20 \
  --deconvolution_identifier MA_bag_identifier
  ```

---

## Performance

The performance of our models was evaluated using two benchmark datasets,
 [CD8 epitopes](#cd8-epitopes) and [MS ligands](#ms-ligands).
These evaluation datasets can be found at:

```bash
gs://bench-mhc/data/netmhcpan-4.1/evaluation/raw_data/
```

The instructions for reproducing the benchmark results shown below are available at [issue #97](https://github.com/instadeepai/BenchMHC/issues/97#issuecomment-2815621149).

### Overall performance table

<!-- markdownlint-disable MD051 -->
<!-- to avoid issues with + in anchor links -->
| Model                                 | HLA                | Non-HLA            | BA                 | SA                 | MA                 | Per-allele Mean Top-K on _MS ligands_ | Per-allele Global Top-K on _MS ligands_| Per-epitope Median FRANK Score on _CD8 epitopes_ | Per-epitope Mean FRANK Score on _CD8 epitopes_ |
| :--------------------- | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :-------------------------------------: | :-------------------------------------: | :-------------------------------------: | :-------------------------------------: |
| NetMHCpan-4.1 - BA Head | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | 0.7537 | 0.7058 | 0.0030 | 0.0114 |
| NetMHCpan-4.1 - EL Head | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | **0.7897** | **0.7887** | **0.0022** | **0.0105** |
| - | - | - | - | - | - | - | - | - | - |
| [(HLA) BA](#hla-ba)                   | :white_check_mark: | :x:                | :white_check_mark: | :x: | :x: |          0.6968       |  0.6304 | 0.0038 | 0.0131 |
| [BA](#ba)           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |  0.701     |   0.6360 | 0.0039 | 0.0132 |
| [(HLA) SA](#hla-sa) | :white_check_mark: | :x: | :x: | :white_check_mark: | :x: |  0.7695     |  0.7599 | 0.0030 | 0.0170 |
| [SA](#sa)           | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :x: |    0.7716   |  0.7638 | 0.0030 | 0.0172 |
| [(HLA) BA + SA](#hla-ba-sa) - BA Head        | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :x: | 0.7374       |  0.6809| 0.0031 | 0.0117 |
| [(HLA) BA + SA](#hla-ba-sa) - EL Head          | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :x: | 0.7815      |  0.774 | 0.0027 | 0.0138 |
| [BA + SA](#ba-sa) - BA Head          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | 0.7460 |  0.6933 | 0.0029 | **0.0113** |
| [BA + SA](#ba-sa) - EL Head| :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |     0.7866  |  0.7773 | 0.0027 | 0.0143 |
| [(HLA) BA + SA + MA](#hla-ba-sa-ma) - BA Head | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |    0.754   |  0.7006 | 0.0030 | 0.0115 |
| [(HLA) BA + SA + MA](#hla-ba-sa-ma) - EL Head   | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |    0.7931   |  0.7866 | **0.0025** | 0.0144 |
| **[BA + SA + MA](#ba-sa-ma)** - BA Head      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |  0.755     |  0.7003 | 0.0030 | 0.0117 |
| **[BA + SA + MA](#ba-sa-ma)** - EL Head      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | **0.7936**      |  **0.7879** | **0.0025** | 0.0138 |
<!-- markdownlint-enable MD051 -->

### Performance plots MS ligands

![Per-allele PPV on MS ligands](../media/models/netmhcpan41_v_0_0_0/ms_ligands_per_allele_box_plot.png#only-light)
![Per-allele PPV on MS ligands](../media/models/netmhcpan41_v_0_0_0/ms_ligands_per_allele_box_plot_dark.png#only-dark)
/// caption
Per-allele PPV on MS ligands for each model. The box plots show the distribution of per-allele
PPVs across the 36 alleles in the evaluation set.
///
![Per-allele PPV on MS ligands](../media/models/netmhcpan41_v_0_0_0/ms_ligands_per_allele_bar_plot.png#only-light)
![Per-allele PPV on MS ligands](../media/models/netmhcpan41_v_0_0_0/ms_ligands_per_allele_bar_plot_dark.png#only-dark)
/// caption
Per-allele PPV on MS ligands for each model. The bar plots show the per-allele PPVs
for each of the 36 alleles in the evaluation set.
///

??? note "MS ligands plots reproduction"

    All metric files required to plot the MS ligands performance plots are available under :arrow_down:
    ```bash
    gs://bench-mhc/data/v0.0.0/evaluation/metrics/metrics_files_ms_ligands/
    ```
    You can simply use the following configuration with the `generate-performance-plot` command line
    to regenerate the above plot.

    ```yaml
    "NetMHCpan-4.1 - BA score":
      metrics_path: "metrics_files_ms_ligands/NetMHCpan_4_1___BA_score.csv"

    "NetMHCpan-4.1 - EL score":
      metrics_path: "metrics_files_ms_ligands/NetMHCpan_4_1___EL_score.csv"
      bright_version_of: "NetMHCpan-4.1 - BA score"

    "BA - BA score":
      metrics_path: "metrics_files_ms_ligands/BA___BA_score.csv"

    "(HLA) BA - BA score":
      metrics_path: "metrics_files_ms_ligands/HLA_BA___BA_score.csv"

    "SA - EL score":
      metrics_path: "metrics_files_ms_ligands/SA___EL_score.csv"

    "(HLA) SA - EL score":
      metrics_path: "metrics_files_ms_ligands/HLA_SA___EL_score.csv"

    "BA+SA - BA score":
      metrics_path: "metrics_files_ms_ligands/BA+SA___BA_score.csv"

    "BA+SA - EL score":
      metrics_path: "metrics_files_ms_ligands/BA+SA___EL_score.csv"
      bright_version_of: "BA+SA - BA score"

    "(HLA) BA+SA - BA score":
      metrics_path: "metrics_files_ms_ligands/HLA_BA+SA___BA_score.csv"

    "(HLA) BA+SA - EL score":
      metrics_path: "metrics_files_ms_ligands/HLA_BA+SA___EL_score.csv"
      bright_version_of: "(HLA) BA+SA - BA score"

    "BA+SA+MA - BA score":
      metrics_path: "metrics_files_ms_ligands/BA+SA+MA___BA_score.csv"

    "BA+SA+MA - EL score":
      metrics_path: "metrics_files_ms_ligands/BA+SA+MA___EL_score.csv"
      bright_version_of: "BA+SA+MA - BA score"

    "(HLA) BA+SA+MA - BA score":
      metrics_path: "metrics_files_ms_ligands/HLA_BA+SA+MA___BA_score.csv"

    "(HLA) BA+SA+MA - EL score":
      metrics_path: "metrics_files_ms_ligands/HLA_BA+SA+MA___EL_score.csv"
      bright_version_of: "(HLA) BA+SA+MA - BA score"
    ```

### Performance plots CD8 epitopes

![Per-epitope FRANK scores on CD8 epitopes](../media/models/netmhcpan41_v_0_0_0/cd8_epitope_frank_scores.png#only-light)
![Per-epitope FRANK scores on CD8 epitopes](../media/models/netmhcpan41_v_0_0_0/cd8_epitope_frank_scores_dark.png#only-dark)
/// caption
Per-epitope FRANK scores on CD8 epitopes for each model. The box plots show
the distribution of FRANK scores across the 1,660 epitopes in the evaluation set.
///

??? note "CD8 epitopes plot reproduction"

    All metric files required to plot the CD8 epitopes performance plot are available under :arrow_down:
    ```bash
    gs://bench-mhc/data/v0.0.0/evaluation/metrics/metrics_files_cd8/
    ```
    You can simply use the following configuration with the `generate-performance-plot` command line
    to regenerate the above plot.

    ```yaml
    "NetMHCpan-4.1 - BA score":
      metrics_path: "metrics_files_cd8/NetMHCpan_4_1___BA_score.csv"

    "NetMHCpan-4.1 - EL score":
      metrics_path: "metrics_files_cd8/NetMHCpan_4_1___EL_score.csv"
      bright_version_of: "NetMHCpan-4.1 - BA score"

    "BA - BA score":
      metrics_path: "metrics_files_cd8/BA___BA_score.csv"

    "(HLA) BA - BA score":
      metrics_path: "metrics_files_cd8/HLA_BA___BA_score.csv"

    "SA - EL score":
      metrics_path: "metrics_files_cd8/SA___EL_score.csv"

    "(HLA) SA - EL score":
      metrics_path: "metrics_files_cd8/HLA_SA___EL_score.csv"

    "BA+SA - BA score":
      metrics_path: "metrics_files_cd8/BA+SA___BA_score.csv"

    "BA+SA - EL score":
      metrics_path: "metrics_files_cd8/BA+SA___EL_score.csv"
      bright_version_of: "BA+SA - BA score"

    "(HLA) BA+SA - BA score":
      metrics_path: "metrics_files_cd8/HLA_BA+SA___BA_score.csv"

    "(HLA) BA+SA - EL score":
      metrics_path: "metrics_files_cd8/HLA_BA+SA___EL_score.csv"
      bright_version_of: "(HLA) BA+SA - BA score"

    "BA+SA+MA - BA score":
      metrics_path: "metrics_files_cd8/BA+SA+MA___BA_score.csv"

    "BA+SA+MA - EL score":
      metrics_path: "metrics_files_cd8/BA+SA+MA___EL_score.csv"
      bright_version_of: "BA+SA+MA - BA score"

    "(HLA) BA+SA+MA - BA score":
      metrics_path: "metrics_files_cd8/HLA_BA+SA+MA___BA_score.csv"

    "(HLA) BA+SA+MA - EL score":
      metrics_path: "metrics_files_cd8/HLA_BA+SA+MA___EL_score.csv"
      bright_version_of: "(HLA) BA+SA+MA - BA score"
    ```

## References

[^1]: [Official DTU page + webserver to make
    predictions](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
[^2]: [_NetMHCpan, a Method for Quantitative Predictions of Peptide Binding to Any HLA-A and -B
    Locus Protein of Known Sequence_, Nielsen et al.,
    2007](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796)
[^3]: [_NetMHCpan-3.0; improved prediction of binding to MHC class I molecules integrating
    information from multiple receptor and peptide length datasets_, Nielsen et al.,
    2016](https://pubmed.ncbi.nlm.nih.gov/27029192/)
[^4]: [_NNAlign: a platform to construct and evaluate artificial neural network models of
    receptor-ligand interactions_, Nielsen et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28407117/)
[^5]: [_NNAlign_MA; MHC peptidome deconvolution for accurate MHC binding motif characterization and
    improved T-cell epitope predictions_, Alvarez et al.,
    2019](https://pubmed.ncbi.nlm.nih.gov/31578220/)
[^6]: [_NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by
    concurrent motif deconvolution and integration of MS MHC eluted ligand data_, Reynisson et al.,
    2020](https://pubmed.ncbi.nlm.nih.gov/32406916/)
[^7]: [_The role of antigen expression in shaping the repertoire of HLA presented ligands_, Alvarez
    et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36060059/)
