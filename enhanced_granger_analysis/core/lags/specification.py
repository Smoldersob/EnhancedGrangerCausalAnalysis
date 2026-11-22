from typing import Literal, TypeAlias


LagSpec: TypeAlias = dict[str, dict[str, tuple[int, ...]]]
LagFeature: TypeAlias = tuple[str, int]
Aggregation: TypeAlias = Literal["union", "intersection", "frequency"]
