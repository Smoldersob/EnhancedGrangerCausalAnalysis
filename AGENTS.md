## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo; domain docs live in the root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Project development rules

### Scope and contracts

- Read `docs/agents/project-context.md` for the validated domain model and current-versus-planned status before changing analysis behavior.
- Treat code as the source of truth for current behavior. Surface documentation conflicts and update relevant docs when behavior or public contracts change.
- Preserve the existing major package structure and module responsibilities by default. Internal refactors are acceptable when behavior and contracts remain stable.
- Public contracts include root exports, documented import paths, configuration schemas, and registry names. Get approval before changing them; when approved, update producers, consumers, boundary tests, and docs together.
- Agents may make reversible, in-scope choices. Ask before changing result semantics, major data contracts, model/removal behavior, or resolving consequential ambiguity.

### Tests and documentation

- Put tests under `enhanced_granger_analysis/tests/`. Test public behavior at the top level and internal behavior under module-named subdirectories. Internal tests supplement public-contract tests; do not bulk-migrate existing tests.
- New test files use the package major/minor suffix (currently `_2_12`).
- Keep docs concise and link to detailed references. Maintain module-specific registry catalogs in `docs/registries/` and the `docs/components_loading.md` overview.
- Keep the public changelog to public-contract changes and registry additions/removals, not routine internal refactors or bug fixes.

### Modeling requirements

- Develop linear methods before nonlinear methods. The first target families are continuous, binary, and count: use the declared type to constrain the family; Poisson is the count default, with negative-binomial as an explicit alternative. Categorical and ordinal targets are deferred.
- Require an explicit base variable type, with per-variable type overrides; do not infer types silently. A multi-output model must have homogeneous target types. Defer mixed-type ensembles.
- Categorical predictors require explicit preprocessing/encoding until native encoding is designed; do not pass category codes through as numeric by default.
- For linear models, target equations have target-specific parameters. Defer the parameter-sharing policy for neural/nonlinear models.
- The main analysis is conditional multivariate: test each source variable's contribution across all outputs, with each target able to use the lagged histories of included variables. Test all selected lags jointly for each source-to-target effect. Preserve output-wise results; keep the classical multivariate test as a reference method.
- Lag-importance detection may be considered later as a separate metric; it is not part of the current scope.
- Require an appropriate statistical test before reporting a statistically supported Granger effect. Predictive fit alone is not evidence of such an effect or proof of intervention-level causality.

### Splitting and optimization

- Hyperparameter optimization requires leakage-safe validation. Keep whole-measurement-segment holdout as a selectable method; add expanding walk-forward folds within each segment, with each segment split chronologically and preprocessing fitted within each training fold. Remove prepared-row random splitters and their aliases. Never carry lag history across segment boundaries.
- Final fitting may use all available data without a validation split; do not describe that fit as independent validation.
- Optimize predictive validation loss/metrics, not Granger p-values or causal-graph recovery. Let the configured objective define aggregation; default to arithmetic means across targets and equal means across walk-forward folds.
- Keep raw p-values. Defer false-discovery-rate adjustment for significance decisions until the underlying tests are calibrated; the benchmark `fdr` score is a separate graph-evaluation metric.
