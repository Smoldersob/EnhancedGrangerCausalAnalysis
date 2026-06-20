import pandas as pd
from sys import argv
import numpy as np


class MetricCalculator:
    def __init__(self, gr_tr_path, pred_path, double_edges=False):
        self.ground_truth = pd.read_csv(gr_tr_path, index_col=0)
        self.predictions = pd.read_csv(pred_path, index_col=0)
        self.double_edges = double_edges

        # Ensure consistent columns and handle potential NaNs before further processing
        # Find common columns to avoid NaN columns if files have differing sets of columns
        common_cols = self.ground_truth.columns.intersection(self.predictions.columns)
        self.ground_truth = self.ground_truth[common_cols]
        self.predictions = self.predictions[common_cols]

        # Ensure consistent indices and handle potential NaNs (e.g. if one has more rows)
        common_idx = self.ground_truth.index.intersection(self.predictions.index)
        self.ground_truth = self.ground_truth.loc[common_idx]
        self.predictions = self.predictions.loc[common_idx]

        # Convert to float to handle potential NaNs (if any introduced by alignment or from source)
        # Then fill NaNs with 0 (as 'no relation') and convert to integer
        self.ground_truth = self.ground_truth.astype(float).fillna(0).astype(int)
        self.predictions = self.predictions.astype(float).fillna(0).astype(int)

        ground_truth = self._to_square_binary(self.ground_truth)
        predictions = self._to_square_binary(self.predictions)

        all_nodes = sorted(
            set(ground_truth.index)
            | set(ground_truth.columns)
            | set(predictions.index)
            | set(predictions.columns)
        )

        ground_truth = ground_truth.reindex(
            index=all_nodes, columns=all_nodes, fill_value=0
        )
        predictions = predictions.reindex(
            index=all_nodes, columns=all_nodes, fill_value=0
        )

        ground_truth = ground_truth.astype(float).fillna(0)
        predictions = predictions.astype(float).fillna(0)

        ground_truth = (ground_truth != 0).astype(int)
        predictions = (predictions != 0).astype(int)

        self.adjusted_gt = ground_truth
        self.adjusted_pred = predictions

        # The subsequent confusion_counts relies on 0/1 integer values
        self.confusion_counts()

    def _to_square_binary(self, df):
        df = df.copy()
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        return df

    def confusion_counts(self):
        pred = self.predictions
        gt = self.ground_truth
        self.tr_pos = int(((pred == 1) & (gt == 1)).to_numpy().sum())
        self.fl_pos = int(((pred == 1) & (gt == 0)).to_numpy().sum())
        self.tr_neg = int(((pred == 0) & (gt == 0)).to_numpy().sum())
        self.fl_neg = int(((pred == 0) & (gt == 1)).to_numpy().sum())

    def fdr(self):
        denom = self.fl_pos + self.tr_pos
        return self.fl_pos/denom if denom > 0 else 0.0

    def tpr(self):
        denom = self.fl_neg + self.tr_pos
        return self.tr_pos/denom if denom > 0 else 0.0

    def fpr(self):
        denom = self.fl_pos + self.tr_neg
        return self.fl_pos/denom if denom > 0 else 0.0

    def shd(self):
        # After the robust __init__, self.predictions and self.ground_truth are guaranteed to be int and aligned
        diff = (self.adjusted_pred - self.adjusted_gt).to_numpy()
        if self.double_edges:
            return int(np.abs(diff).sum())
        else:
            diff = diff + diff.transpose()
            diff[diff > 1] = 1  # Ignoring the double edges.
            return int(np.abs(diff).sum()/2) 

    def accuracy(self):
        total = self.tr_pos + self.fl_pos + self.tr_neg + self.fl_neg
        return (self.tr_pos + self.tr_neg) / total if total > 0 else 0.0

    def precision(self):
        denom = self.tr_pos + self.fl_pos
        return self.tr_pos / denom if denom > 0 else 0.0

    def recall(self):
        denom = self.tr_pos + self.fl_neg
        return self.tr_pos / denom if denom > 0 else 0.0

    def f1(self):
        p = self.precision()
        r = self.recall()
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    def evaluate(self):
        return {
            "tp": self.tr_pos,
            "fp": self.fl_pos,
            "tn": self.tr_neg,
            "fn": self.fl_neg,
            "fdr": self.fdr(),
            "tpr": self.tpr(),
            "fpr": self.fpr(),
            "shd": self.shd(),
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }


if __name__ == '__main__':
    met_cal = MetricCalculator(argv[1], argv[2])
    print(met_cal.evaluate())
