from __future__ import annotations

from src.features.geo_dataset_builder import build_geo_model_dataset
from src.modeling.constrained_mmm import run_constrained_mmm


def main() -> None:
    build_geo_model_dataset()
    artifacts = run_constrained_mmm()
    print("Constrained MMM pipeline complete.")
    print(artifacts["channel_eligibility"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
