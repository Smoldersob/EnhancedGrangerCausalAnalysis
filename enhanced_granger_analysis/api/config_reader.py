# enhanced_granger_analysis/api/config_reader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class ConfigReader:
    """Read configuration documents without constructing analysis components."""

    @classmethod
    def read(cls, path: str | Path) -> dict[str, Any]:
        """Read an analysis configuration and resolve supported file references.

        Args:
            path: Path to a JSON or YAML configuration document.

        Require a mapping at the document root. Resolve supported references
        relative to the containing document, not the process working directory.
        Return plain configuration values for AnalysisConfig.from_mapping().

        Do not normalize backend component names, instantiate selectors or
        components, select a backend, or configure a computation device.
        """
        raise NotImplementedError

    @staticmethod
    def read_document(path: str | Path) -> Any:
        """Read a JSON or YAML document without imposing a root value type.

        Args:
            path: Path to an existing JSON, YML, or YAML file.

        Use safe YAML parsing. Raise the project's configuration or validation
        error for missing files, unsupported extensions, malformed documents,
        and unavailable optional YAML support. Include the source path in errors.
        """
        raise NotImplementedError

    @classmethod
    def resolve_references(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> dict[str, Any]:
        """Resolve explicitly supported file-backed configuration fields.

        Args:
            values: Raw analysis configuration.
            base_dir: Directory of the document containing these values.

        Initially support the existing relations-file convention. Replace that
        reference with the loaded raw document; leave domain parsing to the
        relations parser used by AnalysisConfig.

        Do not interpret arbitrary strings as paths or implement general-purpose
        recursive includes unless the public configuration schema requires them.
        Do not mutate the supplied mapping.
        """
        raise NotImplementedError

    @classmethod
    def read_relations(cls, path: str | Path) -> Any:
        """Read a standalone raw relation declaration document.

        Args:
            path: JSON or YAML file containing supported relation declarations.

        Return the raw document, including a list-valued root when supported.
        Do not compile variable names into feature indices or backend masks.
        """
        return cls.read_document(path)