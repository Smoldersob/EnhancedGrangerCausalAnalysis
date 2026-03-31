from . import constaints, models, initializers, regularizers

# Backward-compatible alias: historical package name is `constaints`.
constraints = constaints

__all__ = [
    "models",
    "initializers",
    "constaints",
    "constraints",
    "regularizers",
]