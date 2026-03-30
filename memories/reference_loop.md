# Why a single-model reference loop is valid

A reference loop can be based on one model instance because consecutive iterations do not require a different architecture, only a different input state (for example: omitted causal variable, different feature mask, or a different constraint configuration).

Key rationale:
- Identical model structure: the number of inputs/outputs and layer types remain the same across iterations.
- Isolation through re-initialization: a full initialize(...) restores the starting state, so parameters from the previous iteration do not leak.
- Deterministic fit step for the current state: each iteration result depends on current data and configuration, not on object history after a proper reset.
- Sufficiency of artifacts: causality analysis requires predictions, errors, and sometimes weights; keeping a separate model object per cause is not necessary.
- Better resource efficiency: fewer object allocations, lower GC overhead, simpler flow, and less helper code.
- Backend-agnostic behavior: this approach is valid as long as the backend respects the state-reset contract during initialize(...).

Correctness conditions:
- initialize(...) must be idempotent and complete (reset of weights, optimizer, masks/constraints, and backend buffers).
- Mutating operations (for example omit_variable) must remain local to the current iteration and be overwritten by the next initialize(...).
- Result aggregation should accept explicit data (predictions/weights) to avoid dependence on model object lifetime.

Used in API components:
- MultiTaskGrangerAPI (reference-model loop).
- GrangerAnalysisResults.update_cause (path with direct predictions/weights).
