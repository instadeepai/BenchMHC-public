"""Define a namespace to compute metrics for binary classification."""

import polars as pl


@pl.api.register_expr_namespace("binary_metrics")
class BinaryMetrics:
    """Custom namespace to compute metrics for binary classification.

    It assumes self is the column with the true labels.

    Attributes:
        _true_labels: True labels i.e. 0 or 1.

    Example:
        ```python
        df.select(top_k=pl.col("y_true").binary_metrics.top_k_score(pl.col("y_proba")))
        ```
    """

    def __init__(self, true_labels: pl.Expr) -> None:
        """Initialize namespace with true labels.

        Args:
            true_labels: True labels i.e. 0 or 1.
        """
        self._true_labels = true_labels

    def true_positives(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the number of true positives.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Number of true positives.
        """
        if sample_weight is None:
            return ((self._true_labels == 1) & (predicted_labels == 1)).sum()
        else:
            return (((self._true_labels == 1) & (predicted_labels == 1)) * sample_weight).sum()

    def true_negatives(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the number of true negatives.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Number of true negatives.
        """
        if sample_weight is None:
            return ((self._true_labels == 0) & (predicted_labels == 0)).sum()
        else:
            return (((self._true_labels == 0) & (predicted_labels == 0)) * sample_weight).sum()

    def false_positives(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the number of false positives.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Number of false positives.
        """
        if sample_weight is None:
            return ((self._true_labels == 0) & (predicted_labels == 1)).sum()
        else:
            return (((self._true_labels == 0) & (predicted_labels == 1)) * sample_weight).sum()

    def false_negatives(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the number of false negatives.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Number of false negatives.
        """
        if sample_weight is None:
            return ((self._true_labels == 1) & (predicted_labels == 0)).sum()
        else:
            return (((self._true_labels == 1) & (predicted_labels == 0)) * sample_weight).sum()

    def accuracy_score(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the accuracy score.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Accuracy score.
        """
        if sample_weight is None:
            return (self._true_labels == predicted_labels).mean()
        else:
            return (
                (self._true_labels == predicted_labels) * sample_weight
            ).sum() / sample_weight.sum()

    def balanced_accuracy_score(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the balanced accuracy score.

        It has the same behavior as sklearn with sample_weight=None and adjusted=False cf
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Balanced accuracy score.
        """
        tp = self.true_positives(predicted_labels, sample_weight)
        tn = self.true_negatives(predicted_labels, sample_weight)

        if sample_weight is None:
            num_neg = (self._true_labels == 0).sum()
            num_pos = self._true_labels.sum()
        else:
            num_neg = ((self._true_labels == 0) * sample_weight).sum()
            num_pos = (self._true_labels * sample_weight).sum()

        tnr = tn / num_neg
        tpr = tp / num_pos

        # Same behavior as sklearn
        # https://github.com/scikit-learn/scikit-learn/blob/2621573e60c295a435c62137c65ae787bf438e61/sklearn/metrics/_classification.py#L2393
        return (
            pl.when(tnr.is_nan()).then(tpr).when(tpr.is_nan()).then(tnr).otherwise((tnr + tpr) / 2)
        )

    def precision_score(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the precision score.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Precision score.
        """
        tp = self.true_positives(predicted_labels, sample_weight)

        if sample_weight is None:
            num_pos_predicted = predicted_labels.sum()
        else:
            num_pos_predicted = (predicted_labels * sample_weight).sum()

        return (tp / num_pos_predicted).fill_nan(0)

    def recall_score(
        self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the recall score.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            Recall score.
        """
        tp = self.true_positives(predicted_labels, sample_weight)

        if sample_weight is None:
            num_pos = self._true_labels.sum()
        else:
            num_pos = (self._true_labels * sample_weight).sum()

        return (tp / num_pos).fill_nan(0)

    def fbeta_score(
        self, predicted_labels: pl.Expr, beta: float, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the F-beta score.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            beta: Determines the weight of recall in the combined score.
            sample_weight: Sample weights.

        Returns:
            F-beta score.
        """
        precision = self.precision_score(predicted_labels, sample_weight)
        recall = self.recall_score(predicted_labels, sample_weight)

        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

        return fbeta.fill_nan(0)

    def f1_score(self, predicted_labels: pl.Expr, sample_weight: pl.Expr | None = None) -> pl.Expr:
        """Compute the F1 score.

        Args:
            predicted_labels: Predicted labels i.e. 0 or 1.
            sample_weight: Sample weights.

        Returns:
            F1 score.
        """
        return self.fbeta_score(predicted_labels, beta=1, sample_weight=sample_weight)

    def average_precision_score(
        self, predicted_probas: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the average precision (AP) score.

        Args:
            predicted_probas: Predicted probabilities i.e. float between 0 and 1.
            sample_weight: Sample weights.

        Returns:
            Average precision (AP) score.
        """
        true_labels_sorted = self._true_labels.sort_by(by=predicted_probas, descending=True)

        if sample_weight is None:
            tp_cum_sum = true_labels_sorted.cum_sum()
            fp_cum_sum = (true_labels_sorted == 0).cum_sum()
            recall = tp_cum_sum / true_labels_sorted.sum()
        else:
            sample_weight_sorted = sample_weight.sort_by(by=predicted_probas, descending=True)
            tp_cum_sum = (true_labels_sorted * sample_weight_sorted).cum_sum()
            fp_cum_sum = ((true_labels_sorted == 0) * sample_weight_sorted).cum_sum()
            recall = tp_cum_sum / (true_labels_sorted * sample_weight_sorted).sum()

        precision = tp_cum_sum / (tp_cum_sum + fp_cum_sum)

        recall_shift = recall.shift(1, fill_value=0)

        return (precision * (recall - recall_shift)).sum()

    def pr_auc_score(
        self, predicted_probas: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the PR-AUC score.

        Args:
            predicted_probas: Predicted probabilities i.e. float between 0 and 1.
            sample_weight: Sample weights.

        Returns:
            PR-AUC score.
        """
        true_labels_sorted = self._true_labels.sort_by(by=predicted_probas, descending=True)

        if sample_weight is None:
            tp_cum_sum = true_labels_sorted.cum_sum()
            fp_cum_sum = (true_labels_sorted == 0).cum_sum()
            recall = tp_cum_sum / true_labels_sorted.sum()
        else:
            sample_weight_sorted = sample_weight.sort_by(by=predicted_probas, descending=True)
            tp_cum_sum = (true_labels_sorted * sample_weight_sorted).cum_sum()
            fp_cum_sum = ((true_labels_sorted == 0) * sample_weight_sorted).cum_sum()
            recall = tp_cum_sum / (true_labels_sorted * sample_weight_sorted).sum()

        # Same logic as sklearn to handle ties in probabilities
        # https://github.com/scikit-learn/scikit-learn/blob/2621573e60c295a435c62137c65ae787bf438e61/sklearn/metrics/_ranking.py#L851
        mask = (predicted_probas.sort(descending=True).diff().fill_null(0) != 0).shift(
            -1, fill_value=True
        )

        precision = tp_cum_sum / (tp_cum_sum + fp_cum_sum)

        precision = precision.filter(mask)
        recall = recall.filter(mask)

        precision_shift = precision.shift(1, fill_value=1)
        recall_shift = recall.shift(1, fill_value=0)

        return (((recall - recall_shift) * (precision + precision_shift)) / 2).sum()

    def roc_auc_score(
        self, predicted_probas: pl.Expr, sample_weight: pl.Expr | None = None
    ) -> pl.Expr:
        """Compute the ROC-AUC score.

        Args:
            predicted_probas: Predicted probabilities i.e. float between 0 and 1.
            sample_weight: Sample weights.

        Returns:
            ROC-AUC score.
        """
        true_labels_sorted = self._true_labels.sort_by(by=predicted_probas, descending=True)

        if sample_weight is None:
            tp_cum_sum = true_labels_sorted.cum_sum()
            fp_cum_sum = (true_labels_sorted == 0).cum_sum()
            num_pos = true_labels_sorted.sum()
            num_neg = true_labels_sorted.len() - num_pos
        else:
            sample_weight_sorted = sample_weight.sort_by(by=predicted_probas, descending=True)
            tp_cum_sum = (true_labels_sorted * sample_weight_sorted).cum_sum()
            fp_cum_sum = ((true_labels_sorted == 0) * sample_weight_sorted).cum_sum()
            num_pos = (true_labels_sorted * sample_weight_sorted).sum()
            num_neg = sample_weight_sorted.sum() - num_pos

        tpr = tp_cum_sum / num_pos
        fpr = fp_cum_sum / num_neg  # codespell:ignore

        # Same logic as sklearn to handle ties in probabilities
        # https://github.com/scikit-learn/scikit-learn/blob/2621573e60c295a435c62137c65ae787bf438e61/sklearn/metrics/_ranking.py#L851
        mask = (predicted_probas.sort(descending=True).diff().fill_null(0) != 0).shift(
            -1, fill_value=True
        )

        tpr = tpr.filter(mask)
        fpr = fpr.filter(mask)  # codespell:ignore

        tpr_shift = tpr.shift(1, fill_value=0)
        fpr_shift = fpr.shift(1, fill_value=0)  # codespell:ignore

        roc_auc = (((fpr - fpr_shift) * (tpr + tpr_shift)) / 2).sum()  # codespell:ignore

        return roc_auc.fill_nan(0)

    def top_k_score(
        self,
        predicted_probas: pl.Expr,
        k: int | pl.Expr | None = None,
        sample_weight: pl.Expr | None = None,
    ) -> pl.Expr:
        """Compute the Top-K score.

        Args:
            predicted_probas: Predicted probabilities i.e. float between 0 and 1.
            k: Optional k to provide. If not provided, k is set to the number of positives.
            sample_weight: Sample weights. Not available for a specific value of k.

        Returns:
            Top-K score.

        Raises:
            ValueError: If sample weights and a specific value of k are provided.
        """
        if isinstance(k, int) and sample_weight is not None:
            raise ValueError(
                "For Top-K computation, if a specific value of k is requested, sample weights "
                "aren't available."
            )

        if k is None:
            k = self._true_labels.sum()

        if sample_weight is None:
            return (
                pl.when(k == 0)
                .then(0)
                .otherwise(
                    self._true_labels.filter(
                        predicted_probas >= predicted_probas.top_k(k=k).min()
                    ).mean()
                )
            )
        else:
            mask_top_k_probas = predicted_probas >= predicted_probas.top_k(k=k).min()

            return (
                pl.when(k == 0)
                .then(0)
                .otherwise(
                    self._true_labels.mul(sample_weight)
                    .filter(mask_top_k_probas)
                    .sum()
                    .truediv(mask_top_k_probas.mul(sample_weight).sum())
                )
            )

    def frank_score(self, predicted_probas: pl.Expr) -> pl.Expr:
        """Compute the Frank score associated with the probabilities.

        We compute the Frank score as the proportion of candidate peptides (decoy) with a higher
        score than the epitope (hit). We also assume that self._true_labels only contains one hit.
        If this is not the case, NaN value is returned.

        Args:
            predicted_probas: Predicted probabilities i.e. float between 0 and 1.

        Returns:
            Frank score.
        """
        num_decoy_candiates = predicted_probas.len() - 1

        hit_prob = predicted_probas.filter(self._true_labels == 1).first()

        num_decoys_higher = (predicted_probas > hit_prob).sum()

        return (
            pl.when(self._true_labels.sum() != 1)
            .then(float("NaN"))
            .otherwise(num_decoys_higher / num_decoy_candiates)
        )
