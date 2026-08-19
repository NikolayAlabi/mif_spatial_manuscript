from pathlib import Path
import copy
import json

MODULE_DIR = Path(
    "/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules"
)

BASE_CONFIG = (
    MODULE_DIR
    / "configs"
    / "stage2a5_interpretable_microcompression.json"
)

OUTPUT_BASE = Path(
    "/projects/ovcare/users/nikolay_alabi/immuno/"
    "stage2_global_modules_v8/stage2a5_microcompression_sensitivity"
)

profiles = {
    "conservative": {
        "semantic_rho": 0.95,
        "residual_rho": 0.98,
        "max_oof_loss": 0.005,
    },
    "balanced": {
        "semantic_rho": 0.90,
        "residual_rho": 0.95,
        "max_oof_loss": 0.01,
    },
    "permissive": {
        "semantic_rho": 0.85,
        "residual_rho": 0.90,
        "max_oof_loss": 0.02,
    },
}

with open(BASE_CONFIG) as f:
    base = json.load(f)

for profile, values in profiles.items():
    cfg = copy.deepcopy(base)

    cfg["profile_name"] = profile
    cfg["output_root"] = str(OUTPUT_BASE / profile)
    cfg["min_pairwise_n"] = 20

    # Use the same thresholds for AR and BT during sensitivity testing.
    for rule in ["state", "metric", "compartment"]:
        cfg[f"{rule}_corr_threshold_by_panel"] = {
            "AR": values["semantic_rho"],
            "BT": values["semantic_rho"],
        }
        cfg[f"{rule}_max_oof_loss_by_panel"] = {
            "AR": values["max_oof_loss"],
            "BT": values["max_oof_loss"],
        }

    cfg["residual_corr_threshold_by_panel"] = {
        "AR": values["residual_rho"],
        "BT": values["residual_rho"],
    }
    cfg["residual_max_oof_loss_by_panel"] = {
        "AR": values["max_oof_loss"],
        "BT": values["max_oof_loss"],
    }

    output_config = (
        MODULE_DIR
        / "configs"
        / f"stage2a5_microcompression_{profile}.json"
    )

    with open(output_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved: {output_config}")