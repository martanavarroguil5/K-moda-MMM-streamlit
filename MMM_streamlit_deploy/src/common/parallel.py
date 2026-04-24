from __future__ import annotations

import os


def configured_n_jobs(task_count: int | None = None) -> int:
    raw_override = os.getenv("MMM_N_JOBS", "").strip()
    cpu_total = os.cpu_count() or 1

    if raw_override in {"-1", "all"}:
        n_jobs = cpu_total
    elif raw_override:
        try:
            n_jobs = max(1, int(raw_override))
        except ValueError:
            n_jobs = max(1, cpu_total - 1)
    else:
        n_jobs = max(1, cpu_total - 1)

    if task_count is not None:
        n_jobs = min(n_jobs, max(1, task_count))
    return max(1, n_jobs)


def parallel_kwargs(task_count: int | None = None, backend: str = "loky") -> dict[str, int | str]:
    return {
        "n_jobs": configured_n_jobs(task_count),
        "backend": backend,
        "batch_size": "auto",
        "pre_dispatch": "2*n_jobs",
    }
