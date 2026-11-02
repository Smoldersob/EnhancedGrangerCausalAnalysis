from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..core.exceptions import DataValidationError


class AnalysisConfigLoader:
	"""Read configuration documents and resolve file references.

	Component declarations remain unchanged. Their interpretation belongs
	to the existing configuration and execution services.
	"""

	@staticmethod
	def read_document(path: str | Path) -> Any:
		"""Read a JSON or YAML document without interpreting its contents."""

		document_path = Path(path).expanduser().resolve()

		if not document_path.is_file():
			raise DataValidationError(
				f"Configuration file does not exist: {document_path}"
			)

		suffix = document_path.suffix.lower()

		if suffix not in {".json", ".yaml", ".yml"}:
			raise DataValidationError(
				f"Unsupported config extension {suffix!r}. "
				"Use .json, .yml or .yaml."
			)

		text = document_path.read_text(encoding="utf-8")

		if suffix == ".json":
			return json.loads(text)

		try:
			import yaml
		except ImportError as exc:
			raise DataValidationError(
				"YAML support requires PyYAML. "
				"Install it with: pip install pyyaml"
			) from exc

		return yaml.safe_load(text)

	@classmethod
	def load_raw_file(
		cls,
		path: str | Path,
	) -> dict[str, Any]:
		"""Read a mapping without resolving embedded file references."""

		document = cls.read_document(path)

		if not isinstance(document, Mapping):
			raise DataValidationError(
				"Top-level configuration must be a mapping/object."
			)

		return dict(document)

	@classmethod
	def load_file(
		cls,
		path: str | Path,
	) -> dict[str, Any]:
		"""Read an analysis mapping and resolve its file references."""

		config_path = Path(path).expanduser().resolve()
		config = cls.load_raw_file(config_path)

		return cls.resolve_references(
			config,
			base_dir=config_path.parent,
		)

	@classmethod
	def resolve_references(
		cls,
		config: Mapping[str, Any],
		*,
		base_dir: str | Path,
	) -> dict[str, Any]:
		"""Resolve supported file references without creating components."""

		resolved = dict(config)
		relations = resolved.get("relations")

		if not isinstance(relations, str):
			return resolved

		if not relations.strip():
			raise DataValidationError(
				"relations file path must be a non-empty string."
			)

		reference = Path(relations).expanduser()

		if not reference.is_absolute():
			reference = Path(base_dir).expanduser().resolve() / reference

		resolved["relations"] = cls.read_document(reference)

		return resolved

	@classmethod
	def load_relations_file(
		cls,
		path: str | Path,
	) -> Any:
		"""Read a relation document without parsing relation declarations."""

		return cls.read_document(path)