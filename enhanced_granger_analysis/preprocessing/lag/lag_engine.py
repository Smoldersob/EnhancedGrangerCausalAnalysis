"""Lag selection, aggregation, overrides, and feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from collections.abc import Sequence
from joblib import Parallel, delayed

from ...core.lag_config import (
    LagEngineConfig,
    LagFeature,
    LagPreparationResult,
    LagSegmentInfo,
    LagSpec,
)
from .registry import LagSelector


def _select_one_dataset(
    selector: LagSelector,
    data: pd.DataFrame,
    outputs: tuple[str, ...],
    dataset_index: int,
) -> LagSpec:
    """Run selection using an independent selector instance."""
    try:
        local_selector = deepcopy(selector)
    except Exception as exc:
        raise TypeError(
            "Lag selectors must support deepcopy so each dataset can "
            "be processed using an independent instance."
        ) from exc

    try:
        return local_selector.fit(data, outputs=outputs)
    except Exception as exc:
        raise RuntimeError(
            f"Lag selection failed for dataset {dataset_index}. Exception: {exc}"
        ) from exc


class LagEngine:
    """Prepare a common lagged feature layout across independent datasets.

    Parallelism is used only for per-dataset selection. Aggregation,
    overrides, layout construction, and feature preparation are sequential.

    The engine does not mutate source DataFrames or the supplied selector.
    """

    def __init__(
        self,
        config: LagEngineConfig | None = None,
        selector: LagSelector | None = None,
        *,
        n_jobs: int = 1,
    ) -> None:
        self.config = deepcopy(
            config if config is not None else LagEngineConfig()
        )
        self.selector = selector

        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, Integral)
            or n_jobs == 0
        ):
            raise ValueError("n_jobs must be a non-zero integer.")

        self.n_jobs = int(n_jobs)

        if selector is not None and not callable(
            getattr(selector, "fit", None)
        ):
            raise TypeError(
                "selector must provide a callable fit method."
            )

        self._validate_config()
    
    def prepare(
        self,
        datasets: Sequence[pd.DataFrame],
        *,
        outputs: Sequence[str],
        predictors: Sequence[str] | None = None,
    ) -> LagPreparationResult:
        """
        Select, merge, override, and prepare lagged data.

        All datasets must have the same named column schema, although
        their column order and number of rows may differ.

        Predictor eligibility is controlled by predictors. Output columns
        are supplied to selectors even if they are not eligible final
        predictors, because selectors require their response values.

        Raises
        ------
        ValueError
            If input data, lag specifications, or overrides are invalid,
            or any dataset has no usable rows after final lagging.
        """
        frames, output_names, predictor_names, selection_names = (
            self._validate_inputs(
                datasets,
                outputs=outputs,
                predictors=predictors,
            )
        )

        predictor_overrides, pair_overrides = (
            self._validate_overrides(
                output_names,
                predictor_names,
            )
        )

        if self.selector is None:
            selection_results: tuple[LagSpec, ...] = ()

            aggregated = self._build_fixed_baseline(
                output_names,
                predictor_names,
            )
        else:
            selection_results = self._select_per_dataset(
                frames,
                output_names,
                predictor_names,
                selection_names,
            )

            aggregated = self._aggregate_selections(
                selection_results,
                output_names,
                predictor_names,
            )

        final_lags = self._apply_overrides(
            aggregated,
            predictor_overrides,
            pair_overrides,
        )

        final_lags = self._validate_lag_spec(
            final_lags,
            output_names,
            predictor_names,
            context="Final lag specification",
        )

        # This ordered description is the common source for data columns,
        # mask columns, and feature metadata.
        feature_pairs = self._build_feature_pairs(
            final_lags,
            output_names,
            predictor_names,
        )

        feature_predictors = tuple(
            predictor
            for predictor, _ in feature_pairs
        )

        feature_lags = np.asarray(
            [lag for _, lag in feature_pairs],
            dtype=np.int64,
        )

        active_set = set(feature_predictors)

        active_predictors = tuple(
            predictor
            for predictor in predictor_names
            if predictor in active_set
        )

        removed_predictors = tuple(
            predictor
            for predictor in predictor_names
            if predictor not in active_set
        )

        block_boundaries = self._build_block_boundaries(
            feature_pairs,
            active_predictors,
        )

        mask = self._build_mask(
            final_lags,
            output_names,
            feature_pairs,
        )

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        segments: list[LagSegmentInfo] = []

        row_offset = 0

        for dataset_index, frame in enumerate(frames):
            segment_X, segment_y, n_drop = self._prepare_segment(
                frame,
                output_names=output_names,
                feature_pairs=feature_pairs,
                dataset_index=dataset_index,
            )

            n_prepared = segment_y.shape[0]

            X_parts.append(segment_X)
            y_parts.append(segment_y)

            segments.append(
                LagSegmentInfo(
                    dataset_index=dataset_index,
                    original_row_count=len(frame),
                    dropped_leading_rows=n_drop,
                    prepared_row_slice=slice(
                        row_offset,
                        row_offset + n_prepared,
                    ),
                    retained_index=frame.index[n_drop:].copy(),
                )
            )

            row_offset += n_prepared

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        return LagPreparationResult(
            X=X,
            y=y,
            output_names=output_names,
            active_predictors=active_predictors,
            removed_predictors=removed_predictors,
            feature_predictors=feature_predictors,
            feature_lags=feature_lags,
            block_boundaries=block_boundaries,
            mask=mask,
            final_lags=final_lags,
            aggregated_lags=deepcopy(aggregated),
            selection_results=selection_results,
            segments=tuple(segments),
        )

    def _validate_config(self) -> None:
        cfg = self.config

        if (
            isinstance(cfg.max_lag, bool)
            or not isinstance(cfg.max_lag, Integral)
            or cfg.max_lag < 0
        ):
            raise ValueError(
                "config.max_lag must be a non-negative integer."
            )

        if not isinstance(cfg.use_lag_zero, bool):
            raise TypeError("config.use_lag_zero must be a bool.")

        if cfg.aggregation not in {
            "union",
            "intersection",
            "frequency",
        }:
            raise ValueError(
                "aggregation must be 'union', 'intersection', "
                "or 'frequency'."
            )

        frequency = cfg.selection_frequency

        if isinstance(frequency, bool):
            raise ValueError(
                "selection_frequency must be in (0, 1]."
            )

        try:
            frequency = float(frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "selection_frequency must be numeric."
            ) from exc

        if not np.isfinite(frequency) or not 0 < frequency <= 1:
            raise ValueError(
                "selection_frequency must be in (0, 1]."
            )

        if not isinstance(cfg.predictor_overrides, Mapping):
            raise TypeError("predictor_overrides must be a mapping.")

        if not isinstance(cfg.pair_overrides, Mapping):
            raise TypeError("pair_overrides must be a mapping.")

    @staticmethod
    def _validate_names(
        values: Sequence[str],
        *,
        label: str,
    ) -> tuple[str, ...]:
        if isinstance(values, str):
            raise TypeError(
                f"{label} must be a sequence of names, not a string."
            )

        names = tuple(values)

        if not names:
            raise ValueError(f"{label} must not be empty.")

        if not all(isinstance(name, str) for name in names):
            raise TypeError(
                f"All {label} entries must be strings."
            )

        if len(names) != len(set(names)):
            raise ValueError(f"{label} contains duplicated names.")

        return names

    def _validate_inputs(
        self,
        datasets: Sequence[pd.DataFrame],
        *,
        outputs: Sequence[str],
        predictors: Sequence[str] | None,
    ) -> tuple[
        tuple[pd.DataFrame, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if isinstance(datasets, pd.DataFrame):
            raise TypeError(
                "datasets must be a sequence of DataFrames, "
                "not one DataFrame."
            )

        frames = tuple(datasets)

        if not frames:
            raise ValueError("datasets must not be empty.")

        schema: tuple[str, ...] | None = None

        for index, frame in enumerate(frames):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Dataset {index} is not a DataFrame."
                )

            if frame.empty:
                raise ValueError(f"Dataset {index} is empty.")

            if not frame.columns.is_unique:
                raise ValueError(
                    f"Dataset {index} has duplicated column names."
                )

            if not all(
                isinstance(name, str)
                for name in frame.columns
            ):
                raise TypeError(
                    f"Dataset {index} column names must be strings."
                )

            if schema is None:
                schema = tuple(frame.columns)
            elif set(frame.columns) != set(schema):
                raise ValueError(
                    f"Dataset {index} has a different column schema."
                )

        assert schema is not None

        output_names = self._validate_names(
            outputs,
            label="outputs",
        )
        predictor_names = self._validate_names(
            schema if predictors is None else predictors,
            label="predictors",
        )

        available = set(schema)

        for label, names in (
            ("outputs", output_names),
            ("predictors", predictor_names),
        ):
            missing = set(names).difference(available)
            if missing:
                raise ValueError(
                    f"Unknown {label}: {sorted(missing)!r}."
                )

        # Preserve requested predictor order, then append target-only
        # columns required by selection.
        selection_names = tuple(
            dict.fromkeys((*predictor_names, *output_names))
        )

        for index, frame in enumerate(frames):
            try:
                values = frame.loc[
                    :, list(selection_names)
                ].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dataset {index} contains non-numeric values "
                    "in selected columns."
                ) from exc

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Dataset {index} contains non-finite values "
                    "in selected columns."
                )

        return (
            frames,
            output_names,
            predictor_names,
            selection_names,
        )

    def _normalize_lags(
        self,
        value: tuple[int, ...],
        *,
        context: str,
    ) -> tuple[int, ...]:
        """Validate canonical explicit lag tuples without filling gaps."""
        if not isinstance(value, tuple):
            raise TypeError(
                f"{context} must be a tuple of explicit lag values."
            )

        if any(
            isinstance(lag, bool)
            or not isinstance(lag, Integral)
            or lag < 0
            for lag in value
        ):
            raise ValueError(
                f"{context} must contain non-negative integers."
            )

        normalized = tuple(int(lag) for lag in value)

        if normalized != tuple(sorted(set(normalized))):
            raise ValueError(
                f"{context} must be sorted and contain no duplicates."
            )

        if not self.config.use_lag_zero and 0 in normalized:
            raise ValueError(
                f"{context} includes lag 0, but use_lag_zero is False."
            )

        return normalized

    def _validate_lag_spec(
        self,
        specification: LagSpec,
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
        *,
        context: str,
    ) -> LagSpec:
        """Validate a complete specification and restore canonical order."""
        if not isinstance(specification, Mapping):
            raise TypeError(f"{context} must be a mapping.")

        if set(specification) != set(outputs):
            raise ValueError(
                f"{context} must contain exactly the requested outputs."
            )

        normalized: LagSpec = {}

        for output in outputs:
            row = specification[output]

            if not isinstance(row, Mapping):
                raise TypeError(
                    f"{context}[{output!r}] must be a mapping."
                )

            if set(row) != set(predictors):
                raise ValueError(
                    f"{context}[{output!r}] must contain exactly "
                    "the expected predictors."
                )

            normalized[output] = {}

            for predictor in predictors:
                lags = self._normalize_lags(
                    row[predictor],
                    context=f"{context}[{output!r}][{predictor!r}]",
                )

                if output == predictor and 0 in lags:
                    raise ValueError(
                        f"{context} contains forbidden own-variable "
                        f"lag 0 for {output!r}."
                    )

                normalized[output][predictor] = lags

        return normalized

    def _validate_overrides(
        self,
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
    ) -> tuple[
        dict[str, tuple[int, ...]],
        dict[str, dict[str, tuple[int, ...]]],
    ]:
        predictor_overrides: dict[str, tuple[int, ...]] = {}
        pair_overrides: dict[
            str,
            dict[str, tuple[int, ...]],
        ] = {}

        for predictor, value in (
            self.config.predictor_overrides.items()
        ):
            if predictor not in predictors:
                raise ValueError(
                    f"Override references unknown predictor "
                    f"{predictor!r}."
                )

            predictor_overrides[predictor] = self._normalize_lags(
                value,
                context=f"predictor_overrides[{predictor!r}]",
            )

        for output, row in self.config.pair_overrides.items():
            if output not in outputs:
                raise ValueError(
                    f"Override references unknown output {output!r}."
                )

            if not isinstance(row, Mapping):
                raise TypeError(
                    f"pair_overrides[{output!r}] must be a mapping."
                )

            pair_overrides[output] = {}

            for predictor, value in row.items():
                if predictor not in predictors:
                    raise ValueError(
                        f"Override references unknown predictor "
                        f"{predictor!r}."
                    )

                lags = self._normalize_lags(
                    value,
                    context=(
                        f"pair_overrides[{output!r}]"
                        f"[{predictor!r}]"
                    ),
                )

                if output == predictor and 0 in lags:
                    raise ValueError(
                        "Pair override explicitly requests forbidden "
                        f"own-variable lag 0 for {output!r}."
                    )

                pair_overrides[output][predictor] = lags

        return predictor_overrides, pair_overrides

    def _build_fixed_baseline(
        self,
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
    ) -> LagSpec:
        positive_lags = tuple(
            range(1, int(self.config.max_lag) + 1)
        )

        return {
            output: {
                predictor: (
                    (0,) + positive_lags
                    if (
                        self.config.use_lag_zero
                        and output != predictor
                    )
                    else positive_lags
                )
                for predictor in predictors
            }
            for output in outputs
        }

    def _select_per_dataset(
        self,
        frames: tuple[pd.DataFrame, ...],
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
        selection_names: tuple[str, ...],
    ) -> tuple[LagSpec, ...]:
        assert self.selector is not None

        raw_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_select_one_dataset)(
                self.selector,
                frame.loc[:, list(selection_names)].copy(),
                outputs,
                dataset_index,
            )
            for dataset_index, frame in enumerate(frames)
        )

        normalized_results: list[LagSpec] = []

        for index, raw_result in enumerate(raw_results):
            validated = self._validate_lag_spec(
                raw_result,
                outputs,
                selection_names,
                context=f"Selector result for dataset {index}",
            )

            # Selector output includes target-only columns, but these
            # are not necessarily eligible final predictors.
            normalized_results.append(
                {
                    output: {
                        predictor: validated[output][predictor]
                        for predictor in predictors
                    }
                    for output in outputs
                }
            )

        return tuple(normalized_results)

    def _aggregate_selections(
        self,
        results: tuple[LagSpec, ...],
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
    ) -> LagSpec:
        if not results:
            raise ValueError(
                "Cannot aggregate an empty collection of selections."
            )

        aggregated: LagSpec = {
            output: {} for output in outputs
        }

        for output in outputs:
            for predictor in predictors:
                lag_sets = [
                    set(result[output][predictor])
                    for result in results
                ]

                if self.config.aggregation == "union":
                    merged = set().union(*lag_sets)

                elif self.config.aggregation == "intersection":
                    merged = lag_sets[0].intersection(
                        *lag_sets[1:]
                    )

                else:
                    candidates = set().union(*lag_sets)
                    threshold = float(
                        self.config.selection_frequency
                    )

                    merged = {
                        lag
                        for lag in candidates
                        if (
                            sum(lag in lags for lags in lag_sets)
                            / len(lag_sets)
                        ) >= threshold
                    }

                aggregated[output][predictor] = tuple(
                    sorted(merged)
                )

        return aggregated

    @staticmethod
    def _apply_overrides(
        baseline: LagSpec,
        predictor_overrides: dict[str, tuple[int, ...]],
        pair_overrides: dict[
            str,
            dict[str, tuple[int, ...]],
        ],
    ) -> LagSpec:
        final = deepcopy(baseline)

        # Predictor-wide replacement, with the self-lag-0 exception.
        for predictor, lags in predictor_overrides.items():
            for output in final:
                final[output][predictor] = (
                    tuple(lag for lag in lags if lag != 0)
                    if output == predictor
                    else lags
                )

        # More specific pair replacements take precedence.
        for output, row in pair_overrides.items():
            for predictor, lags in row.items():
                final[output][predictor] = lags

        return final

    @staticmethod
    def _build_feature_pairs(
        specification: LagSpec,
        outputs: tuple[str, ...],
        predictors: tuple[str, ...],
    ) -> tuple[LagFeature, ...]:
        """Build unique feature identities in deterministic column order."""
        return tuple(
            (predictor, lag)
            for predictor in predictors
            for lag in sorted(
                {
                    lag
                    for output in outputs
                    for lag in specification[output][predictor]
                }
            )
        )

    @staticmethod
    def _build_block_boundaries(
        feature_pairs: tuple[LagFeature, ...],
        active_predictors: tuple[str, ...],
    ) -> np.ndarray:
        """Build endpoints for the ordered active-predictor blocks."""
        widths = {
            predictor: 0
            for predictor in active_predictors
        }

        for predictor, _ in feature_pairs:
            widths[predictor] += 1

        block_widths = np.asarray(
            [
                widths[predictor]
                for predictor in active_predictors
            ],
            dtype=np.int64,
        )

        return np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(block_widths, dtype=np.int64),
            )
        )

    @staticmethod
    def _build_mask(
        specification: LagSpec,
        outputs: tuple[str, ...],
        feature_pairs: tuple[LagFeature, ...],
    ) -> np.ndarray:
        """Build the output-by-feature mask for the exact feature order."""
        feature_indices = {
            feature: column
            for column, feature in enumerate(feature_pairs)
        }

        mask = np.zeros(
            (len(outputs), len(feature_pairs)),
            dtype=bool,
        )

        for output_index, output in enumerate(outputs):
            for predictor, lags in specification[output].items():
                for lag in lags:
                    column = feature_indices[(predictor, lag)]
                    mask[output_index, column] = True

        return mask
    
    @staticmethod
    def _prepare_segment(
        frame: pd.DataFrame,
        *,
        output_names: tuple[str, ...],
        feature_pairs: tuple[LagFeature, ...],
        dataset_index: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Prepare aligned data for one independent dataset."""
        n_drop = max(
            (lag for _, lag in feature_pairs),
            default=0,
        )

        n_source_rows = len(frame)

        if n_source_rows <= n_drop:
            raise ValueError(
                f"Dataset {dataset_index} has {n_source_rows} rows, "
                f"but the final specification requires dropping {n_drop}. "
                "No usable observations remain."
            )

        n_prepared_rows = n_source_rows - n_drop

        X = np.empty(
            (n_prepared_rows, len(feature_pairs)),
            dtype=float,
        )

        # Read each active source variable once.
        predictor_names = tuple(
            dict.fromkeys(
                predictor
                for predictor, _ in feature_pairs
            )
        )

        predictor_values = {
            predictor: frame[predictor].to_numpy(dtype=float)
            for predictor in predictor_names
        }

        for column, (predictor, lag) in enumerate(feature_pairs):
            X[:, column] = predictor_values[predictor][
                n_drop - lag : n_source_rows - lag
            ]

        y = frame.loc[
            :, list(output_names)
        ].to_numpy(dtype=float)[n_drop:].copy()

        return X, y, n_drop