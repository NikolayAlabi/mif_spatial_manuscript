# Stage 4 — matched TURBT→RC change, delta-outcome, and RC-only evaluation

Code directory:

`/projects/ovcare/users/nikolay_alabi/immuno/manuscript/delta_RC_evaluation`

Result directory:

`/projects/ovcare/users/nikolay_alabi/immuno/stage4_delta_rc_evaluation_v1`

## Analysis architecture

### Stage 4A — outcome-blind scoring

Builds two score systems from the frozen Stage2C root modules and frozen Stage2D meta-modules.

**Matched-delta scale**

- fit raw-feature z-score parameters on all eligible TURBT patients in that cohort;
- apply those exact parameters to both TURBT and RC;
- form frozen root-module mean-z scores;
- fit root-module standardization on TURBT and apply it to both sample types before meta-module averaging.

This makes:

`delta = RC - TURBT`

a change measured relative to the pretreatment TURBT distribution.

This intentionally improves on the older matched script, which fitted scaling on stacked TURBT+RC rows.

**RC-only scale**

For RC-only survival analysis, feature and root-module z-score parameters are fit within the RC cohort, analogous to Stage2C's cross-sectional score construction.

No outcomes are used in Stage4A.

### Stage 4B — matched TURBT vs RC change screen

DEG-like paired analysis of every frozen root module and meta-module:

- N matched pairs
- TURBT mean
- RC mean
- mean and median delta
- bootstrap 95% CI for median delta
- paired Wilcoxon P
- paired t-test P
- paired standardized effect size dz
- fraction increasing / decreasing
- BH q-values within each prep-root/meta family
- BH q-values across all programs in panel/cohort

Plots are saved separately by prep root and for meta-modules:

- delta forest
- magnitude-vs-evidence scatter
- top paired spaghetti plots

Primary paired-review flag: `n_pairs >= 10`.
Analyses with >=3 matched pairs can still be reported as exploratory.

### Stage 4C — outcome association of delta

For matched patients only:

**Response**

- any response
- complete response

Delta is tested by univariate logistic regression and repeated OOF AUC.

This should be described as a **response-associated treatment-induced change**, not as a pretreatment predictor, because RC biology is already observed after treatment.

**Survival**

- OS
- RFS

Delta is tested by univariate Cox regression and repeated OOF C-index.

**RC is the primary and only survival time origin in this bundle.**
A delta is not known until RC, so using TURBT-origin survival as the primary clock creates avoidable landmark / immortal-time problems.

No-NAC additionally gets the `no_adj_chemo` sensitivity subset for survival.

### Stage 4D — RC-only survival screen

Cross-sectional RC root modules and meta-modules are evaluated for:

- OS
- RFS

Clinical variables are evaluated univariately in the same contexts:

- Age
- Sex
- ypT / pT
- ypN / pN

Outputs include HRs, 95% CIs, P/q values, repeated OOF C-index, fold SD, direction consistency, and forest plots.

Because RC cohorts can be very small:

- primary: N >=20 and events >=5
- exploratory fitting allowed: N >=8 and events >=3

The status/eligibility flag must always accompany tiny-cohort results.

### Stage 4E — aggregation

Creates:

- all_matched_shift_metrics.csv
- matched_shift_cross_cohort_summary.csv
- all_delta_outcome_metrics.csv
- all_rc_only_metrics.csv

and collaborator-review plot directories separated by prep root and meta-modules.

Outcome plots use the same convention as Stage3B:

- RED = favorable response
- RED = longer survival (negative Cox coefficient / HR<1)
- BLUE = unfavorable

Matched-shift plots are different:
- RED = increased at RC
- BLUE = decreased at RC
and are not inherently favorable/unfavorable.

## Why no RC clinical+spatial combined model yet?

The current TURBT pipeline still has to freeze its Stage3C small-signature selection strategy.

The RC combined model should reuse that exact strategy rather than inventing a second selection framework on much smaller RC cohorts. This bundle therefore completes the RC analogue of Stage3A/3B now; after TURBT Stage3C is frozen, the same modeling rules can be ported to RC.

## Installation

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/delta_RC_evaluation
mkdir -p "$MODULE_DIR/configs"
```

Put all `.py` and `.sh` files in `$MODULE_DIR`.

Put:

`stage4_delta_rc_v1.json`

in `$MODULE_DIR/configs/`.

## Run order

Always wait for one phase to finish before submitting the next.

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/delta_RC_evaluation

MODE=validate       bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
MODE=score_setup    bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
MODE=score_workers  bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"

MODE=shift_setup    bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
MODE=shift_workers  bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"

MODE=delta_setup    bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
MODE=delta_workers  bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"

MODE=rc_setup       bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
MODE=rc_workers     bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"

MODE=aggregate      bash "$MODULE_DIR/submit_stage4_delta_rc_v1.sh"
```

## Completion checks

After score workers:

```bash
ROOT=/projects/ovcare/users/nikolay_alabi/immuno/stage4_delta_rc_evaluation_v1
echo expected=$(awk 'END{print NR-1}' "$ROOT/stage4a_score_worker_index.csv")
echo done=$(find "$ROOT/score_cache" -name .done | wc -l)
```

After shift workers:

```bash
echo expected=$(awk 'END{print NR-1}' "$ROOT/stage4b_shift_worker_index.csv")
echo done=$(find "$ROOT/stage4b_matched_shift" -name .done | wc -l)
```

After delta workers:

```bash
echo expected=$(awk 'END{print NR-1}' "$ROOT/stage4c_delta_worker_index.csv")
echo done=$(find "$ROOT/stage4c_delta_outcomes" -name .done | wc -l)
```

After RC workers:

```bash
echo expected=$(awk 'END{print NR-1}' "$ROOT/stage4d_rc_worker_index.csv")
echo done=$(find "$ROOT/stage4d_rc_only" -name .done | wc -l)
```
