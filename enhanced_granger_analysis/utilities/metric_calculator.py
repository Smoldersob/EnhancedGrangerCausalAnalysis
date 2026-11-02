import pandas as pd
import numpy as np
from sys import argv


class MetricCalculator:
    def __init__(
        self,
        gr_tr_path,
        pred_path,
        double_edges=False,
        evaluate_only_known_gt=True,
        gt_nan_means_unknown=True,
        drop_diagonal=False,
    ):
        self.double_edges = double_edges
        self.evaluate_only_known_gt = evaluate_only_known_gt
        self.gt_nan_means_unknown = gt_nan_means_unknown
        self.drop_diagonal = drop_diagonal

        gt_raw = pd.read_csv(gr_tr_path, index_col=0)
        pred_raw = pd.read_csv(pred_path, index_col=0)

        gt_raw = self._normalize_labels(gt_raw)
        pred_raw = self._normalize_labels(pred_raw)

        self.gt_raw = gt_raw.copy()
        self.pred_raw = pred_raw.copy()

        self.gt_aligned, self.pred_aligned, self.eval_mask = self._align_and_build_mask(
            gt_raw, pred_raw
        )

        self.gt_bin = self._binarize_gt(self.gt_aligned)
        self.pred_bin = self._binarize_pred(self.pred_aligned)

        self.confusion_counts()

    def _normalize_labels(self, df):
        df = df.copy()
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        return df

    def _align_and_build_mask(self, gt, pred):
        all_rows = sorted(set(gt.index) | set(pred.index))
        all_cols = sorted(set(gt.columns) | set(pred.columns))

        gt_aligned = gt.reindex(index=all_rows, columns=all_cols)
        pred_aligned = pred.reindex(index=all_rows, columns=all_cols)

        if self.evaluate_only_known_gt:
            if self.gt_nan_means_unknown:
                eval_mask = gt_aligned.notna()
            else:
                gt_rows = pd.Index(gt.index)
                gt_cols = pd.Index(gt.columns)
                eval_mask = pd.DataFrame(False, index=all_rows, columns=all_cols)
                eval_mask.loc[gt_rows, gt_cols] = True
        else:
            eval_mask = pd.DataFrame(True, index=all_rows, columns=all_cols)

        if self.drop_diagonal and set(all_rows) == set(all_cols):
            common = sorted(set(all_rows) & set(all_cols))
            for node in common:
                if node in eval_mask.index and node in eval_mask.columns:
                    eval_mask.loc[node, node] = False

        gt_aligned = gt_aligned.fillna(0)
        pred_aligned = pred_aligned.fillna(0)

        return gt_aligned, pred_aligned, eval_mask

    def _binarize_gt(self, df):
        return (df.astype(float) != 0).astype(int)

    def _binarize_pred(self, df):
        return (df.astype(float) != 0).astype(int)

    def _masked_arrays(self):
        mask = self.eval_mask.to_numpy(dtype=bool)
        gt = self.gt_bin.to_numpy(dtype=int)[mask]
        pred = self.pred_bin.to_numpy(dtype=int)[mask]
        return gt, pred

    def confusion_counts(self):
        gt, pred = self._masked_arrays()
        self.tr_pos = int(((pred == 1) & (gt == 1)).sum())
        self.fl_pos = int(((pred == 1) & (gt == 0)).sum())
        self.tr_neg = int(((pred == 0) & (gt == 0)).sum())
        self.fl_neg = int(((pred == 0) & (gt == 1)).sum())

    def fdr(self):
        denom = self.fl_pos + self.tr_pos
        return self.fl_pos / denom if denom > 0 else 0.0

    def tpr(self):
        denom = self.fl_neg + self.tr_pos
        return self.tr_pos / denom if denom > 0 else 0.0

    def fpr(self):
        denom = self.fl_pos + self.tr_neg
        return self.fl_pos / denom if denom > 0 else 0.0

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

    def shd(self):
        rev_cost, used = self._reversal_zone_cost()
        ham_cost = self._remaining_hamming_cost(used)
        return int(rev_cost + ham_cost)

    def _reversal_zone_cost(self):
        gt = self.gt_bin
        pred = self.pred_bin
        mask = self.eval_mask

        effects = set(gt.index.astype(str))
        causes = set(gt.columns.astype(str))
        common_nodes = sorted(effects & causes)

        used = pd.DataFrame(False, index=gt.index, columns=gt.columns)
        total = 0

        for i_idx, i in enumerate(common_nodes):
            for j in common_nodes[i_idx + 1:]:
                if i not in gt.index or j not in gt.index:
                    continue
                if i not in gt.columns or j not in gt.columns:
                    continue

                if not (bool(mask.loc[i, j]) and bool(mask.loc[j, i])):
                    continue

                gt_ij = int(gt.loc[i, j])
                gt_ji = int(gt.loc[j, i])
                pr_ij = int(pred.loc[i, j])
                pr_ji = int(pred.loc[j, i])

                if self.double_edges:
                    total += abs(pr_ij - gt_ij) + abs(pr_ji - gt_ji)
                else:
                    if (gt_ij, gt_ji) != (pr_ij, pr_ji):
                        total += 1

                used.loc[i, j] = True
                used.loc[j, i] = True

        return int(total), used


    def _remaining_hamming_cost(self, used):
        remaining_mask = self.eval_mask & (~used)

        gt_arr = self.gt_bin.to_numpy(dtype=int)
        pred_arr = self.pred_bin.to_numpy(dtype=int)
        rem_arr = remaining_mask.to_numpy(dtype=bool)

        diff = (gt_arr != pred_arr)
        return int(diff[rem_arr].sum())

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
            "n_evaluated": int(self.eval_mask.to_numpy(dtype=bool).sum()),
        }


if __name__ == "__main__":
    met_cal = MetricCalculator(argv[1], argv[2])
    print(met_cal.evaluate())