from dataclasses import dataclass


@dataclass(frozen=True)
class VariableRoles:
    """Participation of variables in one concrete Granger analysis."""

    effects: tuple[str, ...]
    causes: tuple[str, ...]
    tested_causes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "causes", tuple(self.causes))
        object.__setattr__(self, "tested_causes", tuple(self.tested_causes))


AnalyticalRoles = VariableRoles
