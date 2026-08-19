# Stage 3A / 3B — TURBT evaluation of frozen root modules and meta-modules

Code location requested:

`/projects/ovcare/users/nikolay_alabi/immuno/manuscript/TURBT_evaluation`

## Scientific scope

Stage 3A evaluates the **frozen Stage2C root modules** and **frozen Stage2D cross-root meta-modules** one at a time against:

- complete response
- any response
- overall survival
- recurrence-free survival

Cohorts:

- NAC2020
- PURE01
- BLASST
- No-NAC
- NAC2015

All evaluations are **TURBT / all patients / median core aggregation / acceptable-or-borderline QC / epithelial fraction >= 0.05**.

NAC2015 is not used to change module membership or meta-module membership. It is scored only after the definitions are frozen.

## IMPORTANT before Stage 3A

The Stage2D files must actually correspond to the frozen rho threshold:

- AR = 0.35
- BT = 0.35

Edit the existing Stage2D config so:

```json
"primary_rho_thresholds": {
  "AR": 0.35,
  "BT": 0.35
}
```

and rerun Stage2D once. Stage3A validation deliberately fails if
`stage2d_primary_meta_summary.csv` still reports 0.30.

## Install

Create:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/TURBT_evaluation
mkdir -p "$MODULE_DIR/configs"
```

Copy:

```text
stage3a_turbt_module_univariate_v1.py
worker_stage3a_nac2015_scores_v1.sh
worker_stage3a_univariate_v1.sh
submit_stage3a_turbt_module_univariate_v1.sh
stage3b_aggregate_turbt_module_results_v1.py
submit_stage3b_aggregate_turbt_module_results_v1.sh
```

to `$MODULE_DIR`, and copy:

```text
stage3a_turbt_module_univariate_v1.json
```

to `$MODULE_DIR/configs/`.

Syntax check:

```bash
python -m py_compile \
  "$MODULE_DIR/stage3a_turbt_module_univariate_v1.py" \
  "$MODULE_DIR/stage3b_aggregate_turbt_module_results_v1.py"

bash -n "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
bash -n "$MODULE_DIR/worker_stage3a_nac2015_scores_v1.sh"
bash -n "$MODULE_DIR/worker_stage3a_univariate_v1.sh"
bash -n "$MODULE_DIR/submit_stage3b_aggregate_turbt_module_results_v1.sh"
```

## Stage 3A run order

Run each mode only after the previous one finishes.

### 1. Validate

```bash
MODE=validate bash "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
```

### 2. Setup

```bash
MODE=setup bash "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
```

Setup reads the existing discovery Stage2C / Stage2D wide score tables once and creates small cohort/panel caches.

### 3. Score NAC2015

```bash
MODE=nac2015 bash "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
```

This submits only two jobs: AR and BT. Each reconstructs NAC2015 scores without using outcomes.

Check:

```bash
ROOT=/projects/ovcare/users/nikolay_alabi/immuno/stage3_turbt_module_evaluation_v1/stage3a_univariate
find "$ROOT/score_cache" -name .done | wc -l
```

After NAC2015 completes there should be 10 completed cohort/panel score caches.

### 4. Finalize the endpoint inventory

```bash
MODE=finalize bash "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
```

Inspect:

```bash
column -s, -t < "$ROOT/stage3a_context_availability_audit.csv" | less -S
```

Contexts with too few events/classes remain in the audit. They are only submitted if a basic univariate fit is possible; they are flagged `primary_eligible=False`.

### 5. Run all univariate context workers

```bash
MODE=workers bash "$MODULE_DIR/submit_stage3a_turbt_module_univariate_v1.sh"
```

One CPU per cohort x panel x endpoint context.

Completion check:

```bash
echo "Expected:"
awk 'END {print NR-1}' "$ROOT/stage3a_context_index.csv"

echo "Completed:"
find "$ROOT/contexts" -name .done | wc -l
```

## Stage 3A metrics

For every frozen program:

- coefficient standardized per 1-SD score increase
- OR or HR + 95% CI
- nominal P
- full-fit AUC / C-index
- repeated OOF AUC / C-index
- mean fold metric
- fold SD
- coefficient-direction consistency across folds
- valid-fold count
- sample / event counts

Repeated CV defaults to 5 repeats of up to 5 folds.

## Stage 3B

After all Stage3A contexts complete:

```bash
bash "$MODULE_DIR/submit_stage3b_aggregate_turbt_module_results_v1.sh"
```

Outputs:

```text
all_stage3a_program_metrics_with_qvalues.csv
program_cross_context_summary.csv
program_response_summary.csv
program_survival_summary.csv
nac2015_program_evaluation_summary.csv
discovery_vs_nac2015_concordance.csv
stage3b_context_inventory.csv
plots/
```

BH corrections are reported both:

1. within root/meta level for each cohort x panel x endpoint; and
2. across all root + meta programs within the context.

Primary recurrence summaries exclude low-signal contexts using the frozen eligibility flag; those contexts remain in the long table for exploratory review.

## Important interpretation of NAC2015

The **spatial-program definitions are frozen** before NAC2015 evaluation.

However, the univariate logistic/Cox coefficient is fitted within NAC2015, so the Stage3A OOF AUC/C-index is best described as:

> independent-cohort evaluation of frozen spatial scores

rather than a fully locked external model.

A future validation cohort can be used for truly locked model transport after the downstream multivariable model is fixed.


## No-NAC no-adjuvant-chemotherapy sensitivity analysis

This bundle now evaluates No-NAC twice for survival endpoints:

```text
No-NAC / OS  / all
No-NAC / OS  / no_adj_chemo
No-NAC / RFS / all
No-NAC / RFS / no_adj_chemo
```

provided each context has enough evaluable patients/events to fit.

The subset definition exactly mirrors Stage1:

```text
no_adj_chemo = adjuvant_chemo normalized to "no"
```

where accepted no-like values are `0`, `0.0`, `no`, `n`, `false`, and `f`.

The frozen module-score cache is **not recomputed** for the subset. The module score is
a biological measurement and does not depend on adjuvant treatment. Stage3A simply
filters the No-NAC patient/outcome rows before fitting the univariate model.

Therefore `all` and `no_adj_chemo` are stored as separate context outputs:

```text
contexts/No-NAC/AR/OS/all/
contexts/No-NAC/AR/OS/no_adj_chemo/
```

and similarly for BT/RFS when evaluable. Stage3B keeps `patient_subset` as an explicit
stratification variable and does not double-count `no_adj_chemo` in the discovery-vs-NAC2015
concordance calculation.


# Revision: clinical univariate screen + manuscript-oriented Stage3B plots

Stage3A now evaluates three explicit program levels:

```text
clinical_variable
root_module
meta_module
```

The prespecified TURBT clinical variables are:

```text
Age
Sex
cT
cN
```

The same candidate-column lists used by the older clinical/module-combined evaluator are used
to locate the usable clean-matrix columns in each cohort.

Every Stage3A context now writes forest plots for:

```text
all variables/programs
clinical variables only
each prep root separately
meta-modules only
```

Thus the forest plots are not deferred to Stage3B.

Stage3B deliberately does NOT put every root module in one giant dotplot.

For each panel, output structure is:

```text
plots/AR/roots/phenotype_only/
plots/AR/roots/AR_state/
plots/AR/roots/AR_checkpoint_state/
plots/AR/roots/compartment/
plots/AR/roots/compartment_state/
plots/AR/meta_modules/
plots/AR/clinical_variables/

plots/BT/roots/phenotype_only/
plots/BT/roots/compartment/
plots/BT/meta_modules/
plots/BT/clinical_variables/
```

Each root/meta directory gets separate:

```text
01_response_effect_dotplot.png
02_survival_effect_dotplot.png
```

Response columns are grouped in this order:

```text
any_response
complete_response
```

and survival columns are grouped:

```text
OS
RFS
```

The effect color convention is fixed:

```text
RED  = favorable
BLUE = unfavorable
```

Specifically:

- response: positive coefficient -> red
- survival: negative Cox coefficient / lower hazard -> red

Therefore the survival plotting function reverses the Cox coefficient sign only for visualization.
The original coefficient and HR are retained unchanged in all result tables.
