# Enhanced Granger Causal Analysis

This project analyzes temporal dependencies among process variables using Granger and Granger-like methods. Its results describe predictive evidence, not proof of intervention-level causation.

## Language

**Granger effect**:
Evidence that the past values of one variable add predictive information about another variable, conditional on the other included variables. The selected lags are tested together; this is predictive evidence, not proof that intervening on the first variable changes the second.
_Avoid_: causal effect (unqualified), proven cause

**Measurement segment**:
A regular, time-ordered measurement window of the same object. Separate segments may have unobserved time between them, and lag history does not continue across their boundaries.
_Avoid_: independent process run

**Walk-forward validation**:
An evaluation approach that trains on earlier observations and tests predictions on later observations, preserving time order within each **Measurement segment**.
