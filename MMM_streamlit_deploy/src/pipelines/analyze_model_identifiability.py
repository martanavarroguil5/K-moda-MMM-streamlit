from __future__ import annotations

from src.modeling.identifiability_audit import run_identifiability_audit


def main() -> None:
    artifacts = run_identifiability_audit()
    print("Identifiability audit complete.")
    print(artifacts["model_comparison"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
