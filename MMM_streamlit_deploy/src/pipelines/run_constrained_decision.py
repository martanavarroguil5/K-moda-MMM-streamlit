from __future__ import annotations

from src.features.geo_dataset_builder import build_geo_model_dataset
from src.modeling.constrained_mmm import run_constrained_mmm
from src.simulation.constrained_decision import run_constrained_decision_layer


def main(planning_budget_eur: float | None = None) -> None:
    build_geo_model_dataset()
    run_constrained_mmm()
    artifacts = run_constrained_decision_layer(planning_budget_eur=planning_budget_eur)
    print("Constrained decision layer complete.")
    print()
    print(artifacts["weights"].round(4).to_string(index=False))
    print()
    print(artifacts["scenarios"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
