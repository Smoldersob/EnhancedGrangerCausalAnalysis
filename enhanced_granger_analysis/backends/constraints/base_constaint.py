from __future__ import annotations

import warnings
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.constraints_config import (
	MinAbsSumRule,
	ProcessedConstraintSpec,
	RelationMap,
	RelationValue,
)
from ...core.exceptions import ConstraintConfigurationError


def _build_feature_blocks(
	col_offsets: Sequence[int],
	n_features: int,
) -> List[NDArray[np.int64]]:
	if len(col_offsets) == 0:
		raise ConstraintConfigurationError("col_offsets cannot be empty")

	offsets = [int(v) for v in col_offsets]
	if offsets[0] != 0:
		raise ConstraintConfigurationError("col_offsets must start at 0")
	if any(offsets[i] > offsets[i + 1] for i in range(len(offsets) - 1)):
		raise ConstraintConfigurationError("col_offsets must be non-decreasing")
	if offsets[-1] > int(n_features):
		raise ConstraintConfigurationError("Last col_offset cannot exceed n_features")

	ends = offsets[1:] + [int(n_features)]
	blocks: List[NDArray[np.int64]] = []
	for start, end in zip(offsets, ends):
		if end < start:
			raise ConstraintConfigurationError("Invalid block boundaries in col_offsets")
		blocks.append(np.arange(start, end, dtype=np.int64))
	return blocks


def _parse_relation_value(value: RelationValue) -> Tuple[str, Optional[float]]:
	"""Parse user relation value into ('zero'|'min_abs_sum', value)."""
	if value is None:
		return "zero", None
	if value is False:
		return "zero", None
	if isinstance(value, str):
		if value.strip().lower() in {"zero", "off", "none", "0"}:
			return "zero", None
		raise ConstraintConfigurationError(f"Unsupported relation string value: {value}")

	if isinstance(value, Mapping):
		if "zero" in value and bool(value["zero"]):
			return "zero", None
		if "min_abs_sum" in value:
			min_sum = float(value["min_abs_sum"])
			if min_sum < 0:
				raise ConstraintConfigurationError("min_abs_sum must be >= 0")
			return ("zero", None) if min_sum == 0 else ("min_abs_sum", min_sum)
		if "force_abs_sum" in value:
			min_sum = float(value["force_abs_sum"])
			if min_sum < 0:
				raise ConstraintConfigurationError("force_abs_sum must be >= 0")
			return ("zero", None) if min_sum == 0 else ("min_abs_sum", min_sum)
		raise ConstraintConfigurationError("Relation dict must include 'zero', 'min_abs_sum', or 'force_abs_sum'")

	min_sum = float(value)
	if min_sum < 0:
		raise ConstraintConfigurationError("Relation numeric value must be >= 0")
	return ("zero", None) if min_sum == 0 else ("min_abs_sum", min_sum)


def process_user_relations(
	relations: RelationMap,
	predictor_names: Sequence[str],
	output_names: Sequence[str],
	col_offsets: Sequence[int],
	n_features: int,
	base_mask: Optional[NDArray[np.float64]] = None,
) -> ProcessedConstraintSpec:
	"""Convert user relation mapping into a tensor-agnostic constraint spec.

	Input format:
	- key: (output_variable_name, input_variable_name)
	- value:
	  - 0 / False / None / 'zero' => zero relation
	  - positive number => enforce minimal sum(abs(weights)) for that relation
	  - {'min_abs_sum': x} => same as above
	 - predictor_names and output_names define the variable order and mapping to indices.
	 - col_offsets and n_features define the structure of the weight matrix and how predictors map to columns
	 - base_mask can be provided to start from an existing mask (e.g. from lag selection) before applying relation rules
	"""
	if len(predictor_names) == 0:
		raise ConstraintConfigurationError("predictor_names cannot be empty")
	if len(output_names) == 0:
		raise ConstraintConfigurationError("output_names cannot be empty")

	blocks = _build_feature_blocks(col_offsets=col_offsets, n_features=n_features)
	if len(blocks) != len(predictor_names):
		raise ConstraintConfigurationError("predictor_names length must match number of blocks from col_offsets")

	pred_idx_map = {name: i for i, name in enumerate(predictor_names)}
	out_idx_map = {name: i for i, name in enumerate(output_names)}

	if base_mask is None:
		mask = np.ones((len(output_names), int(n_features)), dtype=np.float64)
	else:
		mask = np.asarray(base_mask, dtype=np.float64).copy()
		expected_shape = (len(output_names), int(n_features))
		if mask.shape != expected_shape:
			raise ConstraintConfigurationError(f"base_mask shape {mask.shape} does not match {expected_shape}")

	rules: List[MinAbsSumRule] = []
	for (out_name, in_name), raw_value in relations.items():
		if out_name not in out_idx_map:
			raise ConstraintConfigurationError(f"Unknown output variable in relation: {out_name}")
		if in_name not in pred_idx_map:
			raise ConstraintConfigurationError(f"Unknown predictor variable in relation: {in_name}")

		out_idx = out_idx_map[out_name]
		in_idx = pred_idx_map[in_name]
		feature_indices = blocks[in_idx]

		mode, min_abs = _parse_relation_value(raw_value)
		if mode == "zero":
			mask[out_idx, feature_indices] = 0.0
			continue

		assert min_abs is not None
		active = feature_indices[mask[out_idx, feature_indices] > 0.0]
		if active.size == 0:
			warnings.warn(
				f"Skipping min_abs_sum rule for ({out_name}, {in_name}): "
				f"all relation weights are masked to zero (likely due to lag selection). "
				f"This relation will not be constrained.",
				UserWarning,
				stacklevel=3,
			)
			continue
		rules.append(
			MinAbsSumRule(
				output_index=int(out_idx),
				feature_indices=tuple(int(v) for v in active.tolist()),
				min_abs_sum=float(min_abs),
			)
		)

	return ProcessedConstraintSpec(mask=mask, rules=tuple(rules))
